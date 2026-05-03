import argparse
import json
import unittest
from pathlib import Path

from scripts.replay_stable_smc_live_shadow import selected_smc_allocations, selected_smc_case_names
from scripts.smc_short_event_builder import FORMAL_SMC_CASE_NAMES, SMC_CASES


class ReplayStableSmcLiveShadowTests(unittest.TestCase):
    def test_formal_case_and_allocation_parsing(self) -> None:
        args = argparse.Namespace(
            smc_case=None,
            smc_cases="formal",
            smc_allocation=None,
            smc_allocation_values="0.5,1.0",
        )

        self.assertEqual(selected_smc_case_names(args), list(FORMAL_SMC_CASE_NAMES))
        self.assertEqual(selected_smc_allocations(args), [0.5, 1.0])

    def test_single_case_back_compat_overrides_multi_case_args(self) -> None:
        args = argparse.Namespace(
            smc_case="v2_medium_dispbody05_otherlag4_10x",
            smc_cases="v1_base_other_10x",
            smc_allocation=0.75,
            smc_allocation_values="1.0",
        )

        self.assertEqual(selected_smc_case_names(args), ["v2_medium_dispbody05_otherlag4_10x"])
        self.assertEqual(selected_smc_allocations(args), [0.75])

    def test_unknown_smc_case_fails_fast(self) -> None:
        args = argparse.Namespace(
            smc_case=None,
            smc_cases="missing_case",
            smc_allocation=None,
            smc_allocation_values="1.0",
        )

        with self.assertRaisesRegex(ValueError, "Unknown SMC cases"):
            selected_smc_case_names(args)

    def test_all_smc_cases_selects_registered_cases(self) -> None:
        args = argparse.Namespace(
            smc_case=None,
            smc_cases="all",
            smc_allocation=None,
            smc_allocation_values="1.0",
        )

        self.assertEqual(selected_smc_case_names(args), sorted(SMC_CASES))

    def test_existing_replay_report_keeps_selected_candidate_compatible(self) -> None:
        report_path = Path("var/high_leverage_expansion/stable_smc_live_shadow_replay.json")
        if not report_path.exists():
            self.skipTest(f"Replay report not present: {report_path}")
        report = json.loads(report_path.read_text())

        selected = report["selected_candidate"]
        live = report["live_shadow"]
        top = report["candidate_results"][0]

        self.assertEqual(selected["smc_case"], report["metadata"]["smc_case"])
        self.assertEqual(selected["smc_allocation"], report["metadata"]["smc_allocation"])
        self.assertEqual(top["smc_case"], selected["smc_case"])
        self.assertEqual(top["live_shadow"]["total_return_pct"], live["total_return_pct"])
        self.assertEqual(top["reference_gap"], report["reference_gap"])
        self.assertEqual(live["trades"], sum(live["event_type_counts"].values()))
        self.assertEqual(report["candidate_generation"]["smc_summary"]["fee_model"]["roundtrip_cost_pct"], 0.2)
