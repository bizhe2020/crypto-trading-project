from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ops import health_check as hc  # noqa: E402

CFG = dict(hc.DEFAULT_CFG)


def snap(**kw):
    base = {
        "services": {"btc-scalp-standalone": False, "trend-basket": False},
        "positions": [],
        "equity": 6500.0,
        "scalp_log_age_min": None,
    }
    base.update(kw)
    return base


def qqq(mark=660.0, liq=557.0, upl=-379.0):
    return {"instId": "QQQ-USDT-SWAP", "posSide": "long", "pos": 12.42, "upl": upl, "markPx": mark, "liqPx": liq}


def test_orphan_position_when_owner_service_stopped():
    alerts = hc.evaluate(snap(positions=[qqq()]), CFG, peak_equity=6500)
    assert "orphan:QQQ-USDT-SWAP" in alerts


def test_no_orphan_when_owner_running():
    s = snap(positions=[qqq()], services={"btc-scalp-standalone": False, "trend-basket": True})
    assert "orphan:QQQ-USDT-SWAP" not in hc.evaluate(s, CFG, 6500)


def test_expected_active_service_down():
    cfg = {**CFG, "expected_active": ["trend-basket"]}
    assert "down:trend-basket" in hc.evaluate(snap(), cfg, 6500)
    assert "down:trend-basket" not in hc.evaluate(snap(), CFG, 6500)  # 默认不期望任何服务常驻


def test_liq_distance_long_and_short():
    assert round(hc.liq_distance_pct(100, 90, "long"), 1) == 10.0
    assert round(hc.liq_distance_pct(100, 110, "short"), 1) == 10.0
    assert hc.liq_distance_pct(100, 0, "long") is None
    near = qqq(mark=600.0, liq=560.0)  # 6.7% < 8%
    assert "liq:QQQ-USDT-SWAP" in hc.evaluate(snap(positions=[near]), CFG, 6500)
    assert "liq:QQQ-USDT-SWAP" not in hc.evaluate(snap(positions=[qqq()]), CFG, 6500)  # 15.6%


def test_drawdown_reports_highest_crossed_threshold_only():
    a = hc.evaluate(snap(equity=6000.0), CFG, peak_equity=8000.0)  # -25%
    assert [k for k in a if k.startswith("dd:")] == ["dd:25"]
    assert not [k for k in hc.evaluate(snap(equity=7600.0), CFG, 8000.0) if k.startswith("dd:")]  # -5%


def test_stale_log_only_when_scalp_active():
    s = snap(scalp_log_age_min=120.0, services={"btc-scalp-standalone": True, "trend-basket": False})
    assert "stale_log" in hc.evaluate(s, CFG, 6500)
    s["services"]["btc-scalp-standalone"] = False
    assert "stale_log" not in hc.evaluate(s, CFG, 6500)


def test_api_error_is_alert():
    assert "api" in hc.evaluate(snap(api_error="timeout"), CFG, 6500)


def test_renotify_dedupe_and_recovery():
    state: dict = {}
    alerts = {"orphan:QQQ-USDT-SWAP": "x"}
    fire, rec = hc.decide_notifications(alerts, state, now_ts=1000.0, renotify_hours=6)
    assert fire == ["orphan:QQQ-USDT-SWAP"] and rec == []
    fire, _ = hc.decide_notifications(alerts, state, now_ts=1000.0 + 3600, renotify_hours=6)
    assert fire == []  # 6 小时内不重复
    fire, _ = hc.decide_notifications(alerts, state, now_ts=1000.0 + 7 * 3600, renotify_hours=6)
    assert fire == ["orphan:QQQ-USDT-SWAP"]  # 超过 6 小时再提醒
    fire, rec = hc.decide_notifications({}, state, now_ts=1000.0 + 8 * 3600, renotify_hours=6)
    assert fire == [] and rec == ["orphan:QQQ-USDT-SWAP"] and state["active"] == {}


def test_summary_once_per_day_after_hour():
    state: dict = {}
    early = datetime(2026, 11, 20, 0, 30, tzinfo=timezone.utc)
    later = datetime(2026, 11, 20, 1, 5, tzinfo=timezone.utc)
    assert not hc.should_send_summary(state, early, 1)
    assert hc.should_send_summary(state, later, 1)
    state["last_summary_date"] = "2026-11-20"
    assert not hc.should_send_summary(state, later, 1)


def test_owner_mapping_never_matches_unknown():
    assert hc.owner_of("BTC-USDT-SWAP") == "btc-scalp-standalone"
    assert hc.owner_of("XAU-USDT-SWAP") == "trend-basket"
    assert hc.owner_of("ETH-USDT-SWAP") is None
