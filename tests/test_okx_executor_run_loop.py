"""bot/okx_executor.py run_loop 调度的回归测试（模拟时钟，不连交易所、不写真实状态库）。

背景：2026-09-28 关掉 Telegram 后，seconds_until_next_close() 用 int() 截断，循环会在
「收盘 + buffer」前不到 1 秒醒来，latest_closed_candle_time() 仍是上一根，于是再睡整个周期，
导致每隔一根 15m K 线晚 15 分钟处理。这里锁住修复后的行为，并确认 Telegram 已从执行器移除。
"""

from __future__ import annotations

import sys
from datetime import datetime as real_datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bot.okx_executor as ex  # noqa: E402


def _ts(text: str) -> float:
    return real_datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()


@pytest.fixture()
def clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    state = {"t": 0.0}

    class FakeDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return real_datetime.fromtimestamp(state["t"], tz=tz or timezone.utc)

    monkeypatch.setattr(ex, "datetime", FakeDatetime)
    # 每次 sleep 推进模拟时钟（外加 0.5ms 的系统超时）
    monkeypatch.setattr(ex.time, "sleep", lambda seconds: state.__setitem__("t", state["t"] + float(seconds) + 0.0005))
    return state


class _FakeStore:
    def __init__(self, last_processed: str) -> None:
        self.values = {"last_processed_candle_time": last_processed}
        self.actions: list[tuple[str, str, dict]] = []

    def get_value(self, key: str):
        return self.values.get(key)

    def set_value(self, key: str, value) -> None:
        self.values[key] = value

    def append_action(self, timestamp: str, action_type: str, payload: dict) -> None:
        self.actions.append((timestamp, action_type, payload))


def _bare_engine() -> ex.OkxExecutionEngine:
    engine = object.__new__(ex.OkxExecutionEngine)
    engine.config = SimpleNamespace(timeframe="15m", symbol="BTC/USDT:USDT")
    return engine


@pytest.mark.parametrize("frac", [0.0001, 0.3, 0.5, 0.9999])
def test_sleeping_the_returned_wait_always_lands_after_close_buffer(clock, frac):
    engine = _bare_engine()
    clock["t"] = _ts("2026-10-13 07:16:34.000000") + frac  # 刚处理完 07:00 那根
    assert engine.latest_closed_candle_time(5) == "2026-10-13 07:00"
    wait = engine.seconds_until_next_close(5)
    clock["t"] += wait
    # 睡满返回的秒数后，一定已经能看到刚收盘的 07:15 那根
    assert engine.latest_closed_candle_time(5) == "2026-10-13 07:15"


def test_run_loop_processes_every_15m_candle_without_double_wait(clock):
    engine = _bare_engine()
    engine.store = _FakeStore("2026-10-13 07:00")
    engine.bootstrap = lambda: {"status": "ok"}
    processed: list[str] = []

    def fake_evaluate_latest() -> dict:
        clock["t"] += 70.0  # 实盘一次评估（加载数据 + 跑引擎）约 70 秒
        latest = engine.latest_closed_candle_time(5)
        engine.store.set_value("last_processed_candle_time", latest)
        processed.append(latest)
        if len(processed) >= 8:
            raise KeyboardInterrupt
        return {"status": "ok", "processed_candle_time": latest}

    engine.evaluate_latest = fake_evaluate_latest
    clock["t"] = _ts("2026-10-13 07:16:34.400000")
    with pytest.raises(KeyboardInterrupt):
        engine.run_loop(poll_interval_seconds=15, close_buffer_seconds=5)

    assert processed == [
        "2026-10-13 07:15", "2026-10-13 07:30", "2026-10-13 07:45", "2026-10-13 08:00",
        "2026-10-13 08:15", "2026-10-13 08:30", "2026-10-13 08:45", "2026-10-13 09:00",
    ]
    waits = [payload for _, kind, payload in engine.store.actions if kind == "WAIT"]
    assert all(payload["sleep_seconds"] < 900 for payload in waits)  # 不再出现「再睡整 900 秒」的第二次 WAIT
    assert engine.store.actions[-1][1] == "STOP"


def test_executor_config_ignores_legacy_telegram_keys():
    payload = {
        "mode": "paper", "symbol": "BTC/USDT:USDT", "timeframe": "15m", "informative_timeframe": "4h",
        "margin_mode": "isolated", "max_open_positions": 1, "risk_per_trade": 0.035, "leverage": 20.0,
        "state_db_path": "state/unused_test.db",
        # 服务器上的实盘配置还留着这些键：加载时必须被忽略而不是报错
        "telegram_enabled": False, "telegram_token": "<redacted>", "telegram_chat_id": "<redacted>",
        "telegram_poll_interval_seconds": 30,
    }
    cfg = ex.ExecutorConfig.from_dict(payload)
    assert not any(name.startswith("telegram") for name in ex.ExecutorConfig.__dataclass_fields__)
    assert not hasattr(cfg, "telegram_enabled")


def test_executor_module_has_no_telegram_code():
    source = (ROOT / "bot" / "okx_executor.py").read_text(encoding="utf-8").lower()
    assert "telegram" not in source
    assert "import requests" not in source
