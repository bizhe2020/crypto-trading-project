from __future__ import annotations

import unittest

from scripts.report_high_value_frequency_gaps import build_gap_rows


class GapSmcReferenceGapTest(unittest.TestCase):
    def test_build_gap_rows_uses_reference_exit_to_next_entry_windows(self) -> None:
        events = [
            {
                "event_type": "smc_short",
                "entry_time": "2022-05-19 10:15:00+00:00",
                "exit_time": "2022-05-19 10:45:00+00:00",
                "entry_idx": 1,
            },
            {
                "event_type": "smc_short",
                "entry_time": "2022-06-10 21:45:00+00:00",
                "exit_time": "2022-06-11 16:15:00+00:00",
                "entry_idx": 2,
            },
            {
                "event_type": "sota_long",
                "entry_time": "2022-07-18 11:00:00+00:00",
                "exit_time": "2022-07-18 11:45:00+00:00",
                "entry_idx": 3,
            },
        ]

        gaps = build_gap_rows(events, 0.0)

        self.assertEqual(len(gaps), 2)
        self.assertEqual(gaps[0]["gap_start"], "2022-05-19 10:45:00+00:00")
        self.assertEqual(gaps[0]["gap_end"], "2022-06-10 21:45:00+00:00")
        self.assertEqual(gaps[0]["bucket"], "smc_short_cluster")
        self.assertAlmostEqual(gaps[0]["gap_days"], 22.4583)
        self.assertEqual(gaps[1]["gap_start"], "2022-06-11 16:15:00+00:00")
        self.assertEqual(gaps[1]["gap_end"], "2022-07-18 11:00:00+00:00")
        self.assertEqual(gaps[1]["bucket"], "avoid_fill")
        self.assertAlmostEqual(gaps[1]["gap_days"], 36.7812)

    def test_build_gap_rows_classifies_long_only_windows_by_gap_length(self) -> None:
        events = [
            {
                "event_type": "sota_long",
                "entry_time": "2023-01-01 00:00:00+00:00",
                "exit_time": "2023-01-01 00:00:00+00:00",
                "entry_idx": 1,
            },
            {
                "event_type": "sota_long",
                "entry_time": "2023-01-23 12:00:00+00:00",
                "exit_time": "2023-01-23 12:00:00+00:00",
                "entry_idx": 2,
            },
            {
                "event_type": "sota_long",
                "entry_time": "2023-03-01 06:00:00+00:00",
                "exit_time": "2023-03-01 06:00:00+00:00",
                "entry_idx": 3,
            },
            {
                "event_type": "sota_long",
                "entry_time": "2023-04-15 18:00:00+00:00",
                "exit_time": "2023-04-15 18:00:00+00:00",
                "entry_idx": 4,
            },
        ]

        gaps = build_gap_rows(events, 0.0)

        self.assertEqual([gap["bucket"] for gap in gaps], ["secondary_reentry", "low_priority", "avoid_fill"])

    def test_build_gap_rows_respects_minimum_gap_days(self) -> None:
        events = [
            {
                "event_type": "sota_long",
                "entry_time": "2023-01-01 00:00:00+00:00",
                "exit_time": "2023-01-01 00:00:00+00:00",
                "entry_idx": 1,
            },
            {
                "event_type": "sota_long",
                "entry_time": "2023-01-10 00:00:00+00:00",
                "exit_time": "2023-01-10 00:00:00+00:00",
                "entry_idx": 2,
            },
            {
                "event_type": "sota_long",
                "entry_time": "2023-02-15 00:00:00+00:00",
                "exit_time": "2023-02-15 00:00:00+00:00",
                "entry_idx": 3,
            },
        ]

        gaps = build_gap_rows(events, 21.0)

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["gap_start"], "2023-01-10 00:00:00+00:00")
        self.assertEqual(gaps[0]["gap_end"], "2023-02-15 00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
