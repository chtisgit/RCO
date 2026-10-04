import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audit_qwen36_q2_family_controls import (
    FAMILY_ORDER,
    INTERACTION_ORDER,
    SUBFAMILY_ORDER,
    family_for_entry,
    interaction_families_for_entry,
    subfamily_for_entry,
)


class Q2FamilyControlsTest(unittest.TestCase):
    def test_production_manifest_partitions_all_512_search_groups(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "reports/qwen36_35b_base_gguf_manifest.json").read_text())
        counts = {name: 0 for name in FAMILY_ORDER}
        for entry in manifest["entries"]:
            if entry.get("rco_search"):
                counts[family_for_entry(entry)] += 1
        self.assertEqual(counts, {
            "embedding_and_output": 2,
            "linear_attention_projections": 90,
            "self_attention_projections": 40,
            "routed_expert_matrices": 120,
            "shared_expert_matrices": 120,
            "router_and_shared_gate": 80,
            "ssm_alpha_beta": 60,
        })
        self.assertEqual(sum(counts.values()), 512)

    def test_diagnostic_subfamilies_partition_selected_332_groups(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "reports/qwen36_35b_base_gguf_manifest.json").read_text())
        counts = {name: 0 for name in SUBFAMILY_ORDER}
        for entry in manifest["entries"]:
            if not entry.get("rco_search"):
                continue
            family = subfamily_for_entry(entry)
            if family is not None:
                counts[family] += 1
        self.assertEqual(counts, {
            "token_embedding": 1,
            "output_head": 1,
            "linear_attention_qkv": 30,
            "linear_attention_gate": 30,
            "linear_attention_output": 30,
            "routed_expert_down": 40,
            "routed_expert_gate": 40,
            "routed_expert_up": 40,
            "shared_expert_down": 40,
            "shared_expert_gate": 40,
            "shared_expert_up": 40,
        })
        self.assertEqual(sum(counts.values()), 332)

    def test_interaction_family_counts(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "reports/qwen36_35b_base_gguf_manifest.json").read_text())
        counts = {name: 0 for name in INTERACTION_ORDER}
        for entry in manifest["entries"]:
            if entry.get("rco_search"):
                for family in interaction_families_for_entry(entry):
                    counts[family] += 1
        self.assertEqual(counts, {
            "linear_qkv_plus_output": 60,
            "linear_gate_plus_output": 60,
            "routed_down_plus_gate": 80,
        })


if __name__ == "__main__":
    unittest.main()
