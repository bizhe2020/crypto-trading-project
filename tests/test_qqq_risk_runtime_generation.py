from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from scripts.qqq_risk_runtime_generation import append_latest_signal_row
from scripts.refresh_qqq_risk_runtime_inputs import (
    DEFAULT_LONG_CSV,
    DEFAULT_RECENT_CSV,
    DEFAULT_RUNTIME_DIR,
    DEFAULT_SUMMARY,
    ROOT,
    build_market_proxy_macro_file,
    refresh_etf_symbol,
    run_step,
    validate_output_lag,
)


def test_append_latest_signal_row_adds_latest_unlabelled_row() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"], utc=True),
            "target_drawdown": [0.0],
            "future_drawdown_pct": [-1.2],
            "qqq_close": [100.0],
            "raw_prob_10d": [0.12],
            "model_prob_10d": [0.2],
            "fold_train_end": pd.to_datetime(["2024-01-02"], utc=True),
        }
    )
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
            "qqq_close": [100.0, 101.5],
        }
    )
    latest_signal = {
        "date": "2024-01-03",
        "raw_prob_10d": 0.34,
        "model_prob_10d": 0.45,
    }

    out = append_latest_signal_row(
        predictions,
        frame,
        latest_signal,
        fold_train_end=pd.Timestamp("2024-01-02", tz="UTC"),
    )

    assert len(out) == 2
    latest = out.iloc[-1]
    assert latest["date"] == pd.Timestamp("2024-01-03", tz="UTC")
    assert latest["qqq_close"] == pytest.approx(101.5)
    assert latest["raw_prob_10d"] == pytest.approx(0.34)
    assert latest["model_prob_10d"] == pytest.approx(0.45)
    assert latest["fold_train_end"] == pd.Timestamp("2024-01-02", tz="UTC")
    assert pd.isna(latest["target_drawdown"])
    assert pd.isna(latest["future_drawdown_pct"])


def test_append_latest_signal_row_skips_existing_latest_date() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
            "target_drawdown": [0.0, pd.NA],
            "future_drawdown_pct": [-1.2, pd.NA],
            "qqq_close": [100.0, 101.5],
            "raw_prob_10d": [0.12, 0.34],
            "model_prob_10d": [0.2, 0.45],
            "fold_train_end": pd.to_datetime(["2024-01-02", "2024-01-02"], utc=True),
        }
    )
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True), "qqq_close": [100.0, 101.5]})

    out = append_latest_signal_row(
        predictions,
        frame,
        {"date": "2024-01-03", "raw_prob_10d": 0.34, "model_prob_10d": 0.45},
        fold_train_end=pd.Timestamp("2024-01-02", tz="UTC"),
    )

    assert len(out) == 2
    assert list(out["date"]) == list(predictions["date"])


def test_validate_output_lag_uses_reference_time(tmp_path: Path) -> None:
    csv_path = tmp_path / "risk.csv"
    pd.DataFrame({"date": ["2024-01-10"]}).to_csv(csv_path, index=False)

    payload = validate_output_lag(
        csv_path,
        max_lag_days=2,
        reference_time=pd.Timestamp("2024-01-12T12:00:00Z"),
    )

    assert payload["valid"] is True
    assert payload["lag_days"] == 2
    assert payload["reference_time"] == "2024-01-12 12:00:00+00:00"


def test_run_step_records_failure_metadata() -> None:
    summary: dict[str, object] = {"steps": {}}

    with pytest.raises(RuntimeError, match="boom"):
        run_step(summary, "broken", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    payload = summary["steps"]["broken"]
    assert payload["status"] == "error"
    assert "RuntimeError: boom" in payload["error"]


def test_runtime_risk_defaults_do_not_target_frozen_report_artifacts() -> None:
    recent = DEFAULT_RECENT_CSV.relative_to(ROOT)
    long_cycle = DEFAULT_LONG_CSV.relative_to(ROOT)
    summary = DEFAULT_SUMMARY.relative_to(ROOT)

    assert DEFAULT_RECENT_CSV.parent == DEFAULT_RUNTIME_DIR
    assert DEFAULT_LONG_CSV.parent == DEFAULT_RUNTIME_DIR
    assert DEFAULT_SUMMARY.parent == DEFAULT_RUNTIME_DIR
    assert str(recent) == "var/runtime/qqq_risk/qqq_recent_risk_runtime_predictions.csv"
    assert str(long_cycle) == "var/runtime/qqq_risk/qqq_long_cycle_risk_runtime_predictions.csv"
    assert str(summary) == "var/runtime/qqq_risk/qqq_risk_runtime_refresh_summary.json"
    assert "var/reports" not in str(DEFAULT_RECENT_CSV)
    assert "var/reports" not in str(DEFAULT_LONG_CSV)


def test_runtime_qqq_config_uses_runtime_risk_csvs() -> None:
    config_path = Path("config/config.paper.qqq-usdt-aggressive-runtime.json")
    payload = json.loads(config_path.read_text())

    assert payload["recent_risk_predictions_csv"] == "var/runtime/qqq_risk/qqq_recent_risk_runtime_predictions.csv"
    assert payload["long_cycle_risk_predictions_csv"] == "var/runtime/qqq_risk/qqq_long_cycle_risk_runtime_predictions.csv"
    assert payload["recent_risk_predictions_csv"] != "var/reports/qqq_drawdown_lgb_shadow_predictions_macro_subfactor_core.csv"
    assert payload["long_cycle_risk_predictions_csv"] != "var/reports/qqq_long_cycle_correction20d10_qqqonly_lgb_predictions.csv"


def test_frozen_qqq_config_keeps_frozen_report_risk_csvs() -> None:
    config_path = Path("config/config.paper.qqq-usdt-aggressive-frozen.json")
    payload = json.loads(config_path.read_text())

    assert payload["recent_risk_predictions_csv"] == "var/reports/qqq_drawdown_lgb_shadow_predictions_macro_subfactor_core.csv"
    assert payload["long_cycle_risk_predictions_csv"] == "var/reports/qqq_long_cycle_correction20d10_qqqonly_lgb_predictions.csv"


def test_refresh_etf_symbol_retries_later_start_on_http_400(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    class FakeResponse:
        status_code = 400

    def fake_fetch_timeframe(**kwargs):
        start = str(kwargs["start"])
        calls.append(start)
        if start == "1999-01-01T00:00:00Z":
            raise requests.HTTPError("400 Client Error: Bad Request", response=FakeResponse())
        return pd.DataFrame({"date": pd.to_datetime(["2024-01-02"], utc=True)})

    monkeypatch.setattr("scripts.refresh_qqq_risk_runtime_inputs.fetch_timeframe", fake_fetch_timeframe)

    frame, metadata = refresh_etf_symbol(
        session=object(),
        symbol="BTC-USD",
        start="1999-01-01T00:00:00Z",
        end=None,
        output_path=tmp_path / "BTC-USD-1d.feather",
        sleep_seconds=0.0,
        proxy=None,
        timeout_seconds=1.0,
    )

    assert len(frame) == 1
    assert calls == ["1999-01-01T00:00:00Z", "2006-01-01T00:00:00Z"]
    assert metadata["used_start"] == "2006-01-01T00:00:00Z"
    assert metadata["start_fallback_applied"] is True


def test_refresh_etf_symbol_does_not_retry_non_400(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_fetch_timeframe(**kwargs):
        raise RuntimeError("503 upstream unavailable")

    monkeypatch.setattr("scripts.refresh_qqq_risk_runtime_inputs.fetch_timeframe", fake_fetch_timeframe)

    with pytest.raises(RuntimeError, match="503 upstream unavailable"):
        refresh_etf_symbol(
            session=object(),
            symbol="QQQ",
            start="1999-01-01T00:00:00Z",
            end=None,
            output_path=tmp_path / "QQQ-1d.feather",
            sleep_seconds=0.0,
            proxy=None,
            timeout_seconds=1.0,
        )


def test_build_market_proxy_macro_file_uses_uup_and_uso(tmp_path: Path) -> None:
    etf_dir = tmp_path / "etf"
    etf_dir.mkdir(parents=True)
    uup = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True), "close": [28.1, 28.2]})
    uso = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True), "close": [75.0, 76.5]})
    uup.to_feather(etf_dir / "UUP-1d.feather")
    uso.to_feather(etf_dir / "USO-1d.feather")

    output_path = tmp_path / "macro" / "runtime.feather"
    payload = build_market_proxy_macro_file(etf_dir=etf_dir, output_path=output_path)

    frame = pd.read_feather(output_path)
    assert payload["source_mode"] == "market_proxy"
    assert payload["rows"] == 2
    assert list(frame.columns) == ["date", "macro_broad_dollar_index", "macro_wti_oil"]
    assert frame["macro_broad_dollar_index"].tolist() == [28.1, 28.2]
    assert frame["macro_wti_oil"].tolist() == [75.0, 76.5]
