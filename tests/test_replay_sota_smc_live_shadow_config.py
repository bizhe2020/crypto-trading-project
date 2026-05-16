from __future__ import annotations

import argparse
import json
import unittest

from scripts.replay_sota_smc_live_shadow import _apply_config_defaults


def base_args(**overrides: object) -> argparse.Namespace:
    values = {
        "informative_asof_from_15m": None,
        "confirmed_4h_only": None,
        "replay_sync_entry_to_signal_price": None,
        "enable_gap_smc_short": None,
        "enable_sota_score_gate": None,
        "require_non_bearish_structure_for_long": None,
        "enable_long_score_bucket_sizing": None,
        "long_score_bucket_sizing_rules_json": "",
        "gap_smc_case": None,
        "gap_smc_min_flat_days": None,
        "gap_smc_leverage": None,
        "gap_smc_max_stop_distance_pct": None,
        "smc_case": None,
        "smc_allocation": None,
        "smc_min_entry_idx": None,
        "gap_smc_min_entry_idx": None,
        "sota_score_net_min": None,
        "sota_score_bull_min": None,
        "sota_score_bear_max": None,
        "sota_score_conflict_mode": None,
        "stage_trigger_rr_mode": None,
        "time_trailing_rr_mode": None,
        "atr_activation_rr_mode": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class ReplaySotaSmcLiveShadowConfigTest(unittest.TestCase):
    def test_config_defaults_enable_live_replay_switches(self) -> None:
        rules = [{"name": "fvg_near_bear6_target12", "bear_eq": 6}]
        payload = {
            "confirmed_4h_only": True,
            "replay_sync_entry_to_signal_price": True,
            "enable_gap_smc_short_live": True,
            "gap_smc_short_case": "gap_expansion_21d_other_3x",
            "gap_smc_short_min_flat_days": 21.0,
            "gap_smc_short_leverage": 3.0,
            "gap_smc_short_max_stop_distance_pct": 1.5,
            "gap_smc_short_min_entry_idx": 0,
            "smc_min_entry_idx": 500,
            "enable_sota_score_gate_live": True,
            "sota_score_net_min": 3,
            "sota_score_bull_min": 8,
            "sota_score_bear_max": 6,
            "sota_score_conflict_mode": "any",
            "require_non_bearish_structure_for_long_live": True,
            "enable_long_score_bucket_sizing_live": True,
            "long_score_bucket_sizing_rules": rules,
            "stage_trigger_rr_mode": "close",
            "time_trailing_rr_mode": "extreme",
            "atr_activation_rr_mode": "extreme",
        }

        args = _apply_config_defaults(base_args(), payload)

        self.assertTrue(args.confirmed_4h_only)
        self.assertTrue(args.replay_sync_entry_to_signal_price)
        self.assertTrue(args.enable_gap_smc_short)
        self.assertTrue(args.enable_sota_score_gate)
        self.assertTrue(args.require_non_bearish_structure_for_long)
        self.assertTrue(args.enable_long_score_bucket_sizing)
        self.assertEqual(json.loads(args.long_score_bucket_sizing_rules_json), rules)
        self.assertEqual(args.gap_smc_case, "gap_expansion_21d_other_3x")
        self.assertEqual(args.smc_min_entry_idx, 500)
        self.assertEqual(args.gap_smc_min_entry_idx, 0)
        self.assertEqual(args.sota_score_conflict_mode, "any")

    def test_cli_values_override_config_defaults(self) -> None:
        payload = {
            "confirmed_4h_only": False,
            "replay_sync_entry_to_signal_price": False,
            "enable_long_score_bucket_sizing_live": True,
            "long_score_bucket_sizing_rules": [{"name": "from_config"}],
            "gap_smc_short_min_flat_days": 21.0,
            "sota_score_net_min": 3,
        }

        args = _apply_config_defaults(
            base_args(
                confirmed_4h_only=True,
                replay_sync_entry_to_signal_price=True,
            long_score_bucket_sizing_rules_json=json.dumps([{"name": "from_cli"}]),
            gap_smc_min_flat_days=14.0,
            smc_min_entry_idx=200,
            gap_smc_min_entry_idx=300,
            sota_score_net_min=5,
            ),
            payload,
        )

        self.assertTrue(args.confirmed_4h_only)
        self.assertTrue(args.replay_sync_entry_to_signal_price)
        self.assertEqual(json.loads(args.long_score_bucket_sizing_rules_json), [{"name": "from_cli"}])
        self.assertEqual(args.gap_smc_min_flat_days, 14.0)
        self.assertEqual(args.smc_min_entry_idx, 200)
        self.assertEqual(args.gap_smc_min_entry_idx, 300)
        self.assertEqual(args.sota_score_net_min, 5)


if __name__ == "__main__":
    unittest.main()
