"""Metrics shared between the search drivers and the evaluation paths.

Two groups of helpers live here:

  Training-loop losses (gradient enabled):
    compute_reference_log_probs  cache reference log P for KL.
    compute_kl_loss              KL(reference || model) over a batch.
    compute_ce_loss              standard next-token cross-entropy.
    compute_objective            kl / ce dispatcher.

  Post-hoc evaluation metrics (torch.no_grad):
    compute_perplexity           mean perplexity over a tokenized corpus.
    compute_baseline_topk        top-k logit values / indices for the
                                 baseline (full-precision) model.
    compute_all_metrics          NLL + top-k KL of a candidate model
                                 against the baseline topk cache.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from common import get_input_device

logger = logging.getLogger(__name__)

COMPACT_REFERENCE_SCHEMA = 2


def summarize_compact_reference_mass(ref_log_probs, masks=None):
    """Summarize teacher probability retained by a compact top-k cache.

    ``masks`` may be one unshifted calibration tensor or a list of batch
    tensors. Only next-token positions selected by the mask contribute. The
    function streams scalar aggregates and never concatenates cache tensors.
    Returns ``None`` for full-vocabulary or legacy compact caches that do not
    contain retained-mass measurements.
    """
    if not ref_log_probs or not isinstance(ref_log_probs[0], dict):
        return None
    if any("retained_mass" not in item for item in ref_log_probs):
        return None

    if masks is None:
        mask_batches = [None] * len(ref_log_probs)
    elif isinstance(masks, (list, tuple)):
        if len(masks) != len(ref_log_probs):
            raise ValueError("mask batch count does not match reference cache")
        mask_batches = masks
    else:
        mask_batches = []
        offset = 0
        for item in ref_log_probs:
            batch = item["retained_mass"].shape[0]
            mask_batches.append(masks[offset:offset + batch])
            offset += batch
        if offset != masks.shape[0]:
            raise ValueError("mask sample count does not match reference cache")

    total = 0.0
    count = 0
    minimum = float("inf")
    maximum = float("-inf")
    for item, mask in zip(ref_log_probs, mask_batches):
        mass = item["retained_mass"].float()
        if mask is not None:
            selected = mask[:, 1:].to(dtype=torch.bool, device=mass.device)
            if selected.shape != mass.shape:
                raise ValueError(
                    f"shifted mask shape {selected.shape} does not match "
                    f"retained mass shape {mass.shape}")
            mass = mass[selected]
        else:
            mass = mass.reshape(-1)
        if mass.numel() == 0:
            continue
        total += mass.sum().item()
        count += mass.numel()
        minimum = min(minimum, mass.min().item())
        maximum = max(maximum, mass.max().item())

    if count == 0:
        return {
            "token_count": 0,
            "retained_mass_mean": None,
            "retained_mass_min": None,
            "retained_mass_max": None,
            "omitted_mass_mean": None,
            "omitted_mass_max": None,
        }
    mean = total / count
    return {
        "token_count": count,
        "retained_mass_mean": mean,
        "retained_mass_min": minimum,
        "retained_mass_max": maximum,
        "omitted_mass_mean": 1.0 - mean,
        "omitted_mass_max": 1.0 - minimum,
    }


# ----------------------------------------------------------------------------
# Training-loop losses
# ----------------------------------------------------------------------------


@torch.no_grad()
def compute_reference_log_probs(model, calibration_data, batch_size=4,
                                topk=0):
    """Cache next-token reference log-probabilities for the calibration set.

    With topk=0, returns a list of (B, T-1, V) float16 CPU tensors. With
    topk>0, each list item is a schema-versioned dict containing ``values``
    (top-k reference log-probabilities, float16), ``indices`` (int32 token
    IDs), and ``retained_mass`` (the summed teacher probability of those
    entries, float16). The compact form avoids a full-vocabulary CPU cache
    while making the approximation's omitted probability measurable.
    """
    logger.info("Computing reference log-probabilities...")
    device = get_input_device(model)
    ref_log_probs = []
    n = calibration_data.size(0)
    for i in range(0, n, batch_size):
        batch = calibration_data[i:i + batch_size].to(device)
        logits = model(input_ids=batch).logits[:, :-1, :].contiguous().float()
        if topk > 0:
            k = min(topk, logits.size(-1))
            top_values, top_indices = logits.topk(k, dim=-1)
            log_normalizer = logits.logsumexp(dim=-1, keepdim=True)
            top_log_probs = top_values - log_normalizer
            ref_log_probs.append({
                "schema_version": COMPACT_REFERENCE_SCHEMA,
                "values": top_log_probs.half().cpu(),
                "indices": top_indices.to(torch.int32).cpu(),
                "retained_mass": top_log_probs.exp().sum(dim=-1).half().cpu(),
            })
        else:
            ref_log_probs.append(F.log_softmax(logits, dim=-1).half().cpu())
        del logits
    if topk > 0:
        total_tokens = sum(item["values"].numel() // item["values"].size(-1)
                           for item in ref_log_probs)
        mem_bytes = sum(item["values"].nbytes + item["indices"].nbytes
                        + item["retained_mass"].nbytes
                        for item in ref_log_probs)
    else:
        total_tokens = sum(lp.numel() // lp.size(-1) for lp in ref_log_probs)
        mem_bytes = sum(lp.nbytes for lp in ref_log_probs)
    mem_mb = mem_bytes / 1e6
    logger.info(f"Cached reference log-probs: {len(ref_log_probs)} batches, "
                f"{total_tokens} tokens, {mem_mb:.0f} MB, topk={topk}")
    mass = summarize_compact_reference_mass(ref_log_probs)
    if mass is not None:
        logger.info(
            "Teacher top-k retained mass: mean=%.6f min=%.6f max=%.6f; "
            "mean omitted=%.6f max omitted=%.6f",
            mass["retained_mass_mean"], mass["retained_mass_min"],
            mass["retained_mass_max"], mass["omitted_mass_mean"],
            mass["omitted_mass_max"],
        )
    return ref_log_probs


def compute_kl_loss(model, input_ids, ref_log_probs, mask=None, topk=0):
    """KL(reference || model) summed over tokens, normalised over kept tokens.

    Args:
        model: forward returns logits.
        input_ids: (B, T) integer token ids.
        ref_log_probs: (B, T-1, V) reference log-probabilities for the batch.
        mask: (B, T) or None. If provided, marks loss-target token positions
            (e.g. the assistant turn in a chat dataset); only those
            contribute to the loss. If None, every position contributes.
        topk: if > 0, restrict the KL sum to the top-k reference tokens at
            each position (focuses gradient on the most probable tokens and
            avoids the long-tail noise). 0 means full vocabulary.

    Returns:
        Scalar Tensor with gradient enabled.
    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :].contiguous()
    compact_reference = isinstance(ref_log_probs, dict)
    if compact_reference:
        ref_values = ref_log_probs["values"].to(logits.device)
        ref_indices = ref_log_probs["indices"].to(
            device=logits.device, dtype=torch.long)
    else:
        ref_lp = ref_log_probs.to(logits.device)
    shift_mask = None
    if mask is not None:
        shift_mask = mask[:, 1:].contiguous().float().to(logits.device)

    B, T, V = logits.shape
    chunk = 128
    total_kl = 0.0
    total_tokens = 0.0

    for t in range(0, T, chunk):
        te = min(t + chunk, T)
        logits_chunk = logits[:, t:te, :].float()
        if compact_reference:
            tk_idx = ref_indices[:, t:te, :]
            rp_k = ref_values[:, t:te, :].float()
            model_log_z = logits_chunk.logsumexp(dim=-1, keepdim=True)
            mp_k = logits_chunk.gather(-1, tk_idx) - model_log_z
            rp_k_norm = rp_k - rp_k.logsumexp(dim=-1, keepdim=True)
            kl = (rp_k_norm.exp() * (rp_k_norm - mp_k)).sum(dim=-1)
        else:
            model_lp = F.log_softmax(logits_chunk, dim=-1)
            rp = ref_lp[:, t:te, :]

        if not compact_reference and topk > 0 and topk < V:
            _, tk_idx = rp.topk(topk, dim=-1)
            rp_k = rp.gather(-1, tk_idx)
            mp_k = model_lp.gather(-1, tk_idx)
            rp_k_norm = rp_k - rp_k.logsumexp(dim=-1, keepdim=True)
            kl = (rp_k_norm.exp() * (rp_k_norm - mp_k)).sum(dim=-1)
        elif not compact_reference:
            kl = (rp.exp() * (rp - model_lp)).sum(dim=-1)

        if shift_mask is not None:
            m_chunk = shift_mask[:, t:te]
            total_kl = total_kl + (kl * m_chunk).sum()
            total_tokens = total_tokens + m_chunk.sum()
        else:
            total_kl = total_kl + kl.sum()
            total_tokens = total_tokens + kl.numel()

    if isinstance(total_tokens, torch.Tensor):
        total_tokens = total_tokens.clamp(min=1)
    elif total_tokens < 1:
        total_tokens = 1
    return total_kl / total_tokens


def compute_ce_loss(model, input_ids, mask=None):
    """Next-token cross-entropy. mask, if given, selects loss-target positions."""
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous().to(logits.device)
    V = logits.size(-1)
    if mask is not None:
        shift_mask = mask[:, 1:].contiguous().float().to(logits.device)
        per_tok = F.cross_entropy(logits.view(-1, V), labels.view(-1),
                                  reduction='none').view(labels.shape)
        return (per_tok * shift_mask).sum() / shift_mask.sum().clamp(min=1)
    return F.cross_entropy(logits.view(-1, V), labels.view(-1))


def compute_objective(model, input_ids, objective, ref_log_probs=None, mask=None,
                      topk=0):
    """Dispatch to compute_kl_loss or compute_ce_loss by objective name."""
    if objective == 'kl':
        return compute_kl_loss(model, input_ids, ref_log_probs, mask=mask,
                               topk=topk)
    if objective == 'ce':
        return compute_ce_loss(model, input_ids, mask=mask)
    raise ValueError(f"Unknown objective {objective!r}; expected 'kl' or 'ce'.")


# ----------------------------------------------------------------------------
# Post-hoc evaluation metrics
# ----------------------------------------------------------------------------


@torch.no_grad()
def compute_perplexity(model, data, batch_size=1, masks=None):
    """Mean perplexity, variance, and per-sample PPL across data."""
    n = len(data)
    device = next(model.parameters()).device
    use_masks = masks is not None
    nlls: List[float] = []

    for i in trange(0, n, batch_size, desc="Computing perplexity", leave=False):
        j = min(i + batch_size, n)
        x = torch.cat(data[i:j]).to(device)
        logits = model(x).logits[:, :-1, :].contiguous()
        labels = x[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        if use_masks:
            m = torch.cat(masks[i:j]).to(device)[:, 1:].float()
            valid = m.sum(dim=1).clamp(min=1)
            per_seq = (loss * m).sum(dim=1) / valid
        else:
            per_seq = loss.mean(dim=1)
        nlls.extend(per_seq.cpu().tolist())

    mean_nll = np.mean(nlls)
    per_ppl = [np.exp(v) for v in nlls]
    return float(np.exp(mean_nll)), float(np.var(per_ppl)), per_ppl


@torch.no_grad()
def compute_baseline_topk(model, input_ids, batch_size=4, top_k=10, masks=None):
    """Top-k baseline logit values and indices, used as KL reference."""
    model.eval()
    device = next(model.parameters()).device
    vals, idx = [], []
    for i in range(0, input_ids.shape[0], batch_size):
        b = input_ids[i:i + batch_size].to(device)
        logits = model(b).logits[:, :-1, :].float()
        v, k = logits.topk(top_k, dim=-1)
        vals.append(v.cpu()); idx.append(k.cpu())
        del logits; torch.cuda.empty_cache()
    return torch.cat(vals, dim=0), torch.cat(idx, dim=0)


@torch.no_grad()
def compute_all_metrics(model, calibration_data, baseline_topk_vals,
                        baseline_topk_idx, batch_size=4, top_k=10,
                        temperature=1.0, masks=None):
    """Mean NLL and KL-to-baseline (top-k) over the calibration set."""
    model.eval()
    device = next(model.parameters()).device
    use_masks = masks is not None

    total_nll = 0.0
    total_kl = 0.0
    total_tokens = 0
    bidx = 0

    for i in range(0, calibration_data.shape[0], batch_size):
        b = calibration_data[i:i + batch_size].to(device)
        bs = b.shape[0]
        if use_masks:
            shift_masks = masks[i:i + batch_size].to(device)[:, 1:].float()

        logits = model(b).logits
        sl = logits[:, :-1, :].contiguous()
        labels = b[:, 1:].contiguous()

        nll = F.cross_entropy(
            sl.view(-1, sl.size(-1)), labels.view(-1),
            reduction="none",
        ).view(bs, -1)
        if use_masks:
            total_nll += (nll * shift_masks).sum().item()
        else:
            total_nll += nll.sum().item()

        dl = sl.float() / temperature
        vtl = baseline_topk_vals[bidx:bidx + bs].to(device) / temperature
        vti = baseline_topk_idx[bidx:bidx + bs].to(device)

        for s in range(0, dl.shape[1], 256):
            e = min(s + 256, dl.shape[1])
            dlc = dl[:, s:e, :]
            vtlc = vtl[:, s:e, :]
            vtic = vti[:, s:e, :]
            dtl = dlc.gather(dim=-1, index=vtic)
            vtp = F.softmax(vtlc, dim=-1)
            dtp = F.softmax(dtl, dim=-1)
            kl = (vtp * (torch.log(vtp + 1e-10) - torch.log(dtp + 1e-10))).sum(dim=-1)
            if use_masks:
                cm = shift_masks[:, s:e]
                total_kl += (kl * cm).sum().item()
                total_tokens += cm.sum().item()
            else:
                total_kl += kl.sum().item()
                total_tokens += kl.numel()

        bidx += bs
        torch.cuda.empty_cache()

    n = total_tokens if total_tokens > 0 else 1
    return total_nll / n, total_kl / n
