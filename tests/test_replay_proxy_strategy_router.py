from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_proxy_strategy_router import build_btc_path_from_frozen_artifact, equity_from_returns, summarize_path


def test_frozen_btc_daily_return_compounds_all_same_day_events(tmp_path: Path) -> None:
    artifact = {
        "live_shadow": {
            "events": [
                {
                    "decision": "accepted",
                    "entry_time": "2026-01-01T00:00:00+00:00",
                    "exit_time": "2026-01-02T10:00:00+00:00",
                    "return_pct": 10.0,
                    "event_type": "sota_long",
                    "direction": "BULL",
                    "source_effective_leverage": 1.0,
                },
                {
                    "decision": "accepted",
                    "entry_time": "2026-01-01T01:00:00+00:00",
                    "exit_time": "2026-01-02T11:00:00+00:00",
                    "return_pct": 20.0,
                    "event_type": "sota_long",
                    "direction": "BULL",
                    "source_effective_leverage": 1.0,
                },
            ]
        }
    }
    artifact_path = tmp_path / "frozen.json"
    artifact_path.write_text(json.dumps(artifact))

    path, _meta = build_btc_path_from_frozen_artifact(
        frozen_path=artifact_path,
        start=pd.Timestamp("2026-01-01", tz="UTC"),
        end=pd.Timestamp("2026-01-02", tz="UTC"),
        initial_capital=1000.0,
    )

    closed_day = path.loc[path["date"] == pd.Timestamp("2026-01-02", tz="UTC")].iloc[0]
    assert closed_day["btc_equity_raw"] == pytest.approx(1320.0)
    assert closed_day["btc_return"] == pytest.approx(0.32)
    assert equity_from_returns(path["btc_return"], 1000.0).iloc[-1] == pytest.approx(1320.0)


def test_summarize_path_includes_first_day_switch_cost() -> None:
    path = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-04", "2026-03-05"], utc=True),
            "router_equity": [999.0, 1100.0],
            "router_return": [0.0, 1100.0 / 999.0 - 1.0],
        }
    )

    summary = summarize_path(path, "router_equity", "router_return", initial_capital=1000.0)

    assert summary["total_return_pct"] == pytest.approx(10.0)
    assert summary["max_drawdown_pct"] == pytest.approx(0.1)
    assert summary["annual_returns_pct"]["2026"] == pytest.approx(10.0)
