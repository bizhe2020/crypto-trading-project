from __future__ import annotations

import unittest

from scripts.sota_long_filters import apply_sota_structure_gate


class SotaLongStructureGateTest(unittest.TestCase):
    def test_disabled_gate_keeps_all_candidates(self) -> None:
        events = [
            {"feature_bearish_structure": False},
            {"feature_bearish_structure": True},
        ]
        filtered, diagnostics = apply_sota_structure_gate(events, enabled=False)
        self.assertEqual(len(filtered), 2)
        self.assertFalse(diagnostics["enabled"])
        self.assertEqual(diagnostics["removed_candidates"], 0)

    def test_enabled_gate_removes_bearish_structure_longs(self) -> None:
        events = [
            {"feature_bearish_structure": False, "entry_idx": 1},
            {"feature_bearish_structure": True, "entry_idx": 2},
            {"feature_bearish_structure": False, "entry_idx": 3},
        ]
        filtered, diagnostics = apply_sota_structure_gate(events, enabled=True)
        self.assertEqual([event["entry_idx"] for event in filtered], [1, 3])
        self.assertTrue(diagnostics["enabled"])
        self.assertEqual(diagnostics["removed_candidates"], 1)
        self.assertEqual(diagnostics["filtered_candidates"], 2)


if __name__ == "__main__":
    unittest.main()
