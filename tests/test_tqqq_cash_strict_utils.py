from __future__ import annotations

import pandas as pd

from scripts.tqqq_cash_strict_utils import build_hard_exit_reset_mask


def test_build_hard_exit_reset_mask_accepts_percent_threshold() -> None:
    frame = pd.DataFrame(
        {
            "entry_signal": [1] * 20,
            "ixic_close": [100.0] * 15 + [98.0, 97.0, 96.0, 98.0, 99.0],
            "qqq_close": [100.0] * 20,
            "tqqq_close": [100.0] * 20,
        }
    )
    allow_mask = pd.Series([True] * len(frame))

    mask = build_hard_exit_reset_mask(frame, "ixic_mom15_ge_-2.5%", allow_mask)

    assert bool(mask.iloc[15]) is True
    assert bool(mask.iloc[16]) is False


def test_build_hard_exit_reset_mask_main_desired_matches_entry_and_allow() -> None:
    frame = pd.DataFrame(
        {
            "entry_signal": [1, 1, 0, 1],
            "ixic_close": [100.0, 101.0, 102.0, 103.0],
            "qqq_close": [100.0, 101.0, 102.0, 103.0],
            "tqqq_close": [100.0, 101.0, 102.0, 103.0],
        }
    )
    allow_mask = pd.Series([True, False, True, True])

    mask = build_hard_exit_reset_mask(frame, "main_desired", allow_mask)

    assert mask.tolist() == [True, False, False, True]
