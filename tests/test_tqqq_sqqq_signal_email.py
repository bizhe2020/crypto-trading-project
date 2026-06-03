from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from scripts.send_tqqq_sqqq_signal_email import refresh_market_data


def test_refresh_market_data_retries_direct_when_proxy_fetch_fails(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_output_path_for(output_dir: Path, symbol: str, timeframe: str) -> Path:
        return output_dir / f"{symbol}-{timeframe}.feather"

    def fake_fetch_timeframe(*, session, symbol, timeframe, start, end, output_path, sleep_seconds, proxy, timeout_seconds=60.0):
        calls.append((symbol, proxy))
        if proxy:
            raise RuntimeError("proxy unavailable")
        return pd.DataFrame({"date": pd.to_datetime(["2026-06-02T13:30:00Z"], utc=True)})

    monkeypatch.setattr("scripts.send_tqqq_sqqq_signal_email.output_path_for", fake_output_path_for)
    monkeypatch.setattr("scripts.send_tqqq_sqqq_signal_email.fetch_timeframe", fake_fetch_timeframe)

    refresh_market_data(
        {
            "data_root": str(tmp_path),
            "timeframe": "1d",
            "refresh_proxy": "http://127.0.0.1:7892",
            "refresh_symbols": ["QQQ"],
            "refresh_start": "2022-01-01T00:00:00Z",
            "refresh_sleep_seconds": 0.0,
            "refresh_fail_open": False,
        }
    )

    assert calls == [("QQQ", "http://127.0.0.1:7892"), ("QQQ", None)]


def test_refresh_market_data_still_raises_when_proxy_and_direct_fail(monkeypatch, tmp_path: Path) -> None:
    def fake_output_path_for(output_dir: Path, symbol: str, timeframe: str) -> Path:
        return output_dir / f"{symbol}-{timeframe}.feather"

    def fake_fetch_timeframe(*, session, symbol, timeframe, start, end, output_path, sleep_seconds, proxy, timeout_seconds=60.0):
        if proxy:
            raise RuntimeError("proxy unavailable")
        raise requests.HTTPError("direct failed")

    monkeypatch.setattr("scripts.send_tqqq_sqqq_signal_email.output_path_for", fake_output_path_for)
    monkeypatch.setattr("scripts.send_tqqq_sqqq_signal_email.fetch_timeframe", fake_fetch_timeframe)

    try:
        refresh_market_data(
            {
                "data_root": str(tmp_path),
                "timeframe": "1d",
                "refresh_proxy": "http://127.0.0.1:7892",
                "refresh_symbols": ["QQQ"],
                "refresh_start": "2022-01-01T00:00:00Z",
                "refresh_sleep_seconds": 0.0,
                "refresh_fail_open": False,
            }
        )
    except requests.HTTPError as exc:
        assert "direct failed" in str(exc)
    else:
        raise AssertionError("Expected direct refresh failure to propagate")
