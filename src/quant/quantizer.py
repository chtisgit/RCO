from typing import Iterable, Dict, List, Any, Optional, Union


import os
import re
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


from common import to, maybe_first_element
from quant import dist_utils
from quant.model_utils import InputCollector, ForwardInterrupt, LINEAR_LAYERS, select_layers
from quant.quant_utils import QLinear
from quant.qparams import save_qparams

from quant.fast_obq import FastOBQ


class Quantizer:

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        quantizable_modules: str,
        pre_block_modules: List[str],
        save_dir: Union[str, os.PathLike],
        block_modules: str,
        obq_kwargs: Dict[str, Any] = {},
        device: Optional[torch.device] = None,
        cpu_offload_modules: bool = False,
        cpu_offload_activations: bool = False,
        save_fake_quant: bool = False,
        expert_chunk_size: int = 16,
        checkpoint_loader=None,
        min_free_bytes: int = 0,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.quantizable_modules = quantizable_modules
        self.pre_block_modules = pre_block_modules
        self.block_modules = block_modules
        self.save_dir = save_dir
        self.obq_kwargs = obq_kwargs
        self.device = device
        self.cpu_offload_modules = cpu_offload_modules
        self.cpu_offload_activations = cpu_offload_activations
        self.save_fake_quant = save_fake_quant
        if expert_chunk_size < 1:
            raise ValueError("expert_chunk_size must be positive")
        self.expert_chunk_size = expert_chunk_size
        self.checkpoint_loader = checkpoint_loader
        self.min_free_bytes = int(min_free_bytes)
        self.verbose = verbose

    @torch.no_grad()
    def quantize(self, bitwidth_options: List[int], calibration_bitwidth: int):
        device = self.device or next(self.model.parameters()).device
        # prepare pre blocks modules
        blocks = self._get_submodule(self.block_modules)
        pre_blocks = [self._get_submodule(module_name) for module_name in self.pre_block_modules]
        if self.checkpoint_loader is not None:
            self.checkpoint_loader.move_runtime_buffers(self.model, device)
            for module_name in self.pre_block_modules:
                self.checkpoint_loader.load_prefix(
                    self.model, module_name, device=device)
            self.checkpoint_loader.load_prefix(
                self.model, f"{self.block_modules}.0", device=device)
        else:
            blocks[0] = blocks[0].to(device)
            for module in pre_blocks:
                module.to(device)
        # Cache
        if hasattr(self.model.config, "use_cache"):
            use_cache = self.model.config.use_cache
            self.model.config.use_cache = False
        # Input preparation #
        blocks[0] = InputCollector(blocks[0], cpu_offload=self.cpu_offload_activations)
        # TODO make namedtuple
        for inp_args, inp_kwargs in self.data_loader:
            try:
                self.model(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            except ForwardInterrupt:
                pass
        input_args = blocks[0].input_args
        input_kwargs = blocks[0].input_kwargs
        blocks[0] = blocks[0].module

        if dist_utils.is_dist_available_and_initialized():
            dist.barrier()

        # offload pre_blocks
        if self.checkpoint_loader is not None:
            for module_name in self.pre_block_modules:
                self.checkpoint_loader.release_prefix(self.model, module_name)
        elif self.cpu_offload_modules:
            for module in pre_blocks:
                module.cpu()

        # Block pruning #
        for block_id, block in enumerate(blocks):
            # TODO change to logging
            if self.verbose:
                dist_utils.print_on_main(f"Processing {self.block_modules} {block_id}/{len(blocks)}.")
            block_path = f"{self.block_modules}.{block_id}"
            if self.checkpoint_loader is not None:
                if block_id > 0:
                    self.checkpoint_loader.load_prefix(
                        self.model, block_path, device=device)
                block = self._get_submodule(block_path)
            else:
                block = block.to(device)
            # get layer prefix to select layers only within the block
            layer_prefix = f"{self.block_modules}.{block_id}."
            layers = select_layers(self.model, layer_prefix, self.quantizable_modules, LINEAR_LAYERS)
            handles, hooks = self._prepare_hooks_and_handles(bitwidth_options, layers)

            targets = []
            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))

            for _, h in hooks.items():
                h.remove()

            if dist_utils.is_dist_available_and_initialized():
                dist.barrier()

            self._quant_group(handles, bitwidth_options, calibration_bitwidth)

            self._quantize_fused_experts(
                block,
                input_args,
                input_kwargs,
                bitwidth_options,
                calibration_bitwidth,
                device,
            )

            for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))  # me
                out = maybe_first_element(out)
                if self.cpu_offload_activations:
                    out = out.cpu()
                # change only first input argument
                if len(inp_args) > 0:
                    inp_args[0].data = out
                elif "hidden_states" in inp_kwargs:
                    inp_kwargs["hidden_states"] = out
                else:
                    raise ValueError("Unsupported block input format.")

            if self.checkpoint_loader is not None:
                self.checkpoint_loader.release_prefix(self.model, block_path)
            elif self.cpu_offload_modules:
                block = block.cpu()

            del handles
            del hooks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = use_cache

    def _get_submodule(self, module_name: str):
        return self.model.get_submodule(module_name)

    def _prepare_hooks_and_handles(self, bitwidth_options: List[int], layers: Dict[str, nn.Module]):
        handles = {}
        hooks = {}
        for layer_name, layer in layers.items():

            def update_handle_hook(name):
                def _hook(_, inp, out):
                    handles[name].update(inp[0])

                return _hook

            handles[layer_name] = self._create_handle(bitwidth_options, layer)
            hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))
        return handles, hooks

    def _create_handle(self, bitwidth_options, layer):
        return FastOBQ(layer, bitwidth_options=bitwidth_options, **self.obq_kwargs)

    def _quant_group(self, handles: Dict[str, FastOBQ], bitwidth_options: List[int], calibration_bitwidth: int):
        for handle_name, handle in handles.items():
            self._quantize_handle(
                handle_name, handle, bitwidth_options,
                calibration_bitwidth, replace_module=True)

    def _quantize_handle(
        self,
        handle_name: str,
        handle: FastOBQ,
        bitwidth_options: List[int],
        calibration_bitwidth: int,
        *,
        replace_module: bool,
    ) -> torch.Tensor:
        """Quantize, persist, and return the calibration candidate weight."""
        if self.verbose:
            dist_utils.print_on_main(f"Quantizing {handle_name}")
        self._check_free_space()
        qweight_dict, scale_dict, zero_dict, perm = handle.quantize(
            bitwidth_options)
        calibration_weight = None

        for bits in bitwidth_options:
            layer_save_dir = os.path.join(self.save_dir, handle_name)
            os.makedirs(layer_save_dir, exist_ok=True)
            save_qparams(
                layer_save_dir,
                bits=int(bits),
                qweight=qweight_dict[bits],
                scales=scale_dict[bits],
                zeros=zero_dict[bits],
                perm=perm,
                handle=handle,
            )
            needs_qlayer = self.save_fake_quant or bits == calibration_bitwidth
            if needs_qlayer:
                qlayer = QLinear(
                    qweight_dict[bits],
                    scale_dict[bits],
                    zero_dict[bits],
                    bias=handle.layer.bias,
                    perm=perm,
                    bits=8 if bits > 4 else 4,
                )
                dequantized_weight = qlayer.get_weight()
            if self.save_fake_quant:
                torch.save(
                    dequantized_weight.cpu(),
                    os.path.join(layer_save_dir, f"{int(bits)}.pth"),
                )
            if bits == calibration_bitwidth:
                calibration_weight = dequantized_weight
                if replace_module:
                    parent_name, child_name = handle_name.rsplit(".", 1)
                    parent_module = self.model.get_submodule(parent_name)
                    setattr(parent_module, child_name, qlayer)

        handle.reset()
        if calibration_weight is None:
            raise RuntimeError(
                f"Calibration bitwidth {calibration_bitwidth} was not quantized")
        return calibration_weight

    def _check_free_space(self) -> None:
        if self.min_free_bytes <= 0:
            return
        path = self.save_dir
        while not os.path.exists(path):
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
        free = shutil.disk_usage(path).free
        if free < self.min_free_bytes:
            raise RuntimeError(
                f"Stopping before writing another candidate: filesystem for "
                f"{self.save_dir} has {free / 2**30:.1f} GiB free, below the "
                f"configured floor of {self.min_free_bytes / 2**30:.1f} GiB")

    @staticmethod
    def _linear_view(weight: torch.Tensor) -> nn.Linear:
        """Create a Linear whose parameter aliases one 2-D fused-weight slice."""
        layer = nn.Linear(
            weight.shape[1], weight.shape[0], bias=False,
            device="meta", dtype=weight.dtype)
        layer.weight = nn.Parameter(weight.detach(), requires_grad=False)
        return layer

    @staticmethod
    def _find_fused_experts(block: nn.Module):
        for relative_name, module in block.named_modules():
            gate_up = getattr(module, "gate_up_proj", None)
            down = getattr(module, "down_proj", None)
            if (isinstance(gate_up, (nn.Parameter, torch.Tensor))
                    and isinstance(down, (nn.Parameter, torch.Tensor))
                    and gate_up.ndim == 3 and down.ndim == 3):
                yield relative_name, module

    @torch.no_grad()
    def _quantize_fused_experts(
        self,
        block: nn.Module,
        input_args,
        input_kwargs,
        bitwidth_options: List[int],
        calibration_bitwidth: int,
        device,
    ) -> None:
        """Quantize fused experts in bounded groups without unfusing the model."""
        block_prefix = None
        for name, candidate in self.model.named_modules():
            if candidate is block:
                block_prefix = name
                break
        if block_prefix is None:
            raise ValueError("Could not resolve transformer block path")

        for relative_name, experts in self._find_fused_experts(block):
            experts_path = f"{block_prefix}.{relative_name}".rstrip(".")
            gate_up = experts.gate_up_proj
            down = experts.down_proj
            n_experts, fused_intermediate, hidden = gate_up.shape
            intermediate = fused_intermediate // 2
            expected_down = (n_experts, hidden, intermediate)
            if tuple(down.shape) != expected_down:
                raise ValueError(
                    f"Unexpected {experts_path}.down_proj shape "
                    f"{tuple(down.shape)}; expected {expected_down}")

            for first in range(0, n_experts, self.expert_chunk_size):
                indices = list(range(
                    first, min(first + self.expert_chunk_size, n_experts)))
                records = {}
                for expert_index in indices:
                    prefix = f"{experts_path}.{expert_index}"
                    names = {
                        "gate": f"{prefix}.gate_proj",
                        "up": f"{prefix}.up_proj",
                        "down": f"{prefix}.down_proj",
                    }
                    enabled = {
                        key: re.search(self.quantizable_modules, name) is not None
                        for key, name in names.items()
                    }
                    if not any(enabled.values()):
                        continue
                    gate_weight = gate_up[expert_index, :intermediate, :]
                    up_weight = gate_up[expert_index, intermediate:, :]
                    down_weight = down[expert_index]
                    records[expert_index] = {
                        "names": names,
                        "enabled": enabled,
                        "gate_weight": gate_weight,
                        "up_weight": up_weight,
                        "down_weight": down_weight,
                        "input_handle": (
                            self._create_handle(
                                bitwidth_options, self._linear_view(gate_weight))
                            if enabled["gate"] or enabled["up"] else None
                        ),
                        "down_handle": (
                            self._create_handle(
                                bitwidth_options, self._linear_view(down_weight))
                            if enabled["down"] else None
                        ),
                    }
                if not records:
                    continue

                def collect_inputs(_module, args, kwargs):
                    hidden_states = (args[0] if args else
                                     kwargs.get("hidden_states"))
                    selected = (args[1] if len(args) > 1 else
                                kwargs.get("top_k_index"))
                    if hidden_states is None or selected is None:
                        raise ValueError(
                            f"Cannot extract hidden states and routing indices "
                            f"from {experts_path} forward inputs")
                    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
                    flat_selected = selected.reshape(flat_hidden.shape[0], -1)
                    for expert_index, record in records.items():
                        token_index = torch.where(flat_selected == expert_index)[0]
                        if token_index.numel() == 0:
                            continue
                        current = flat_hidden[token_index]
                        if record["input_handle"] is not None:
                            record["input_handle"].update(current)
                        gate = F.linear(current, record["gate_weight"])
                        up = F.linear(current, record["up_weight"])
                        act_fn = getattr(experts, "act_fn", F.silu)
                        intermediate_states = act_fn(gate) * up
                        if record["down_handle"] is not None:
                            record["down_handle"].update(intermediate_states)

                hook = experts.register_forward_pre_hook(
                    collect_inputs, with_kwargs=True)
                try:
                    for args, kwargs in zip(input_args, input_kwargs):
                        block(*to(args, device=device), **to(kwargs, device=device))
                finally:
                    hook.remove()

                for expert_index, record in records.items():
                    input_handle = record["input_handle"]
                    down_handle = record["down_handle"]
                    missing_input = (input_handle is not None
                                     and input_handle.H is None)
                    missing_down = (down_handle is not None
                                    and down_handle.H is None)
                    if missing_input or missing_down:
                        raise RuntimeError(
                            f"Expert {expert_index} in {experts_path} received no "
                            "calibration tokens; increase or diversify calibration data")

                    # Preserve one pristine input Hessian. Gate and up consume it
                    # sequentially, so only one additional copy is live.
                    if input_handle is not None:
                        input_hessian = input_handle.H
                        input_samples = input_handle.num_samples
                        for key, target_weight in (
                            ("gate", record["gate_weight"]),
                            ("up", record["up_weight"]),
                        ):
                            if not record["enabled"][key]:
                                continue
                            layer = self._linear_view(target_weight)
                            handle = self._create_handle(bitwidth_options, layer)
                            handle.H = input_hessian.clone()
                            handle.num_samples = input_samples
                            calibrated = self._quantize_handle(
                                record["names"][key], handle, bitwidth_options,
                                calibration_bitwidth, replace_module=False)
                            target_weight.copy_(calibrated.to(target_weight.dtype))

                    if record["enabled"]["down"]:
                        calibrated = self._quantize_handle(
                            record["names"]["down"], down_handle,
                            bitwidth_options, calibration_bitwidth,
                            replace_module=False)
                        record["down_weight"].copy_(
                            calibrated.to(record["down_weight"].dtype))
                    if input_handle is not None:
                        input_handle.reset()

                del records
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
