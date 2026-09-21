"""BTC scalp 优化新增能力的单元测试。

覆盖：
  - 波动率目标仓位的缩放系数（含裁剪、边界与关闭时的恒等行为）
  - ATR 止损宽度钳制（上限收窄 / 下限放宽 / 关闭）
  - ExecutorConfig -> StrategyConfig 的新字段映射
  - 影子风控闸门新语义（暂停天数 / 不重置 peak / 永久停牌）

所有测试都不依赖外部数据。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.okx_executor import ExecutorConfig  # noqa: E402
from strategy.scalp_robust_v2_core import (  # noqa: E402
    Candle,
    Direction,
    ScalpRobustEngine,
    StrategyConfig,
    align_timeframes,
    build_precomputed_state_confirmed_4h,
)


def _make_candles(count: int = 400, base: float = 100.0, half_range: float = 1.0) -> list[Candle]:
    """构造恒定 true range = 2*half_range 的 15m K 线，便于精确断言 ATR。"""
    candles: list[Candle] = []
    ts = 1_600_000_000.0
    for i in range(count):
        close = base + (i % 3) * 0.1
        candles.append(
            Candle(
                ts=ts + i * 900.0,
                o=close,
                h=close + half_range,
                l=close - half_range,
                c=close,
                v=10.0,
            )
        )
    return candles


def _make_engine(config: StrategyConfig, count: int = 400, half_range: float = 1.0) -> ScalpRobustEngine:
    c15m = _make_candles(count=count, half_range=half_range)
    # 4h K 线与 15m 同频构造即可，测试只用到 15m 侧的 ATR 与入场辅助函数。
    c4h = [
        Candle(ts=c15m[i * 16].ts, o=c15m[i * 16].o, h=c15m[i * 16].h, l=c15m[i * 16].l, c=c15m[i * 16].c, v=1.0)
        for i in range(count // 16)
    ]
    mapping = align_timeframes(c4h, c15m)
    precomputed = build_precomputed_state_confirmed_4h(c4h, c15m)
    return ScalpRobustEngine(c4h, c15m, mapping, precomputed, config)


def test_vol_target_scale_identity_when_disabled() -> None:
    engine = _make_engine(StrategyConfig(enable_vol_target_sizing=False, vol_target_reference_pct=0.25))
    assert engine._vol_target_scale(300) == 1.0


def test_vol_target_scale_inverse_to_realized_vol() -> None:
    # ATR = 2.0，价格 ~100 → realized ≈ 2.0%；reference 0.5% → 0.25
    config = StrategyConfig(
        enable_vol_target_sizing=True,
        vol_target_reference_pct=0.5,
        vol_target_min_scale=0.05,
        vol_target_max_scale=4.0,
    )
    engine = _make_engine(config)
    scale = engine._vol_target_scale(300)
    realized = engine._atr_for_idx(300) / engine.c15m[300].c * 100.0
    assert abs(scale - 0.5 / realized) < 1e-6


def test_vol_target_scale_is_clipped() -> None:
    low = _make_engine(
        StrategyConfig(
            enable_vol_target_sizing=True,
            vol_target_reference_pct=0.01,
            vol_target_min_scale=0.25,
            vol_target_max_scale=2.0,
        )
    )
    assert low._vol_target_scale(300) == 0.25
    high = _make_engine(
        StrategyConfig(
            enable_vol_target_sizing=True,
            vol_target_reference_pct=50.0,
            vol_target_min_scale=0.25,
            vol_target_max_scale=2.0,
        )
    )
    assert high._vol_target_scale(300) == 2.0


def test_vol_target_scale_insufficient_data_returns_one() -> None:
    engine = _make_engine(StrategyConfig(enable_vol_target_sizing=True, vol_target_reference_pct=0.5))
    # idx 越界 → ATR 取不到 → 恒等
    assert engine._vol_target_scale(10_000) == 1.0


def test_atr_stop_bounds_disabled_keeps_structure_stop() -> None:
    engine = _make_engine(StrategyConfig(enable_atr_stop_bounds=False, atr_stop_cap_multiple=3.5))
    stop, info = engine._apply_atr_stop_bounds(300, Direction.BULL, 100.0, 90.0)
    assert stop == 90.0
    assert info is None


def test_atr_stop_bounds_cap_narrows_wide_stop() -> None:
    # ATR = 2.0，结构止损距离 10 = 5×ATR > cap 3.5×ATR → 收窄到 7
    engine = _make_engine(
        StrategyConfig(enable_atr_stop_bounds=True, atr_stop_cap_multiple=3.5),
    )
    atr = engine._atr_for_idx(300)
    assert atr > 0
    stop, info = engine._apply_atr_stop_bounds(300, Direction.BULL, 100.0, 100.0 - 5.0 * atr)
    assert info is not None and info["bound"] == "cap"
    assert abs((100.0 - stop) - 3.5 * atr) < 1e-9


def test_atr_stop_bounds_floor_widens_tight_stop() -> None:
    engine = _make_engine(
        StrategyConfig(enable_atr_stop_bounds=True, atr_stop_floor_multiple=2.0),
    )
    atr = engine._atr_for_idx(300)
    stop, info = engine._apply_atr_stop_bounds(300, Direction.BEAR, 100.0, 100.0 + 0.5 * atr)
    assert info is not None and info["bound"] == "floor"
    assert abs((stop - 100.0) - 2.0 * atr) < 1e-9


def test_atr_stop_bounds_no_change_inside_band() -> None:
    engine = _make_engine(
        StrategyConfig(enable_atr_stop_bounds=True, atr_stop_floor_multiple=2.0, atr_stop_cap_multiple=3.5),
    )
    atr = engine._atr_for_idx(300)
    original = 100.0 - 2.5 * atr
    stop, info = engine._apply_atr_stop_bounds(300, Direction.BULL, 100.0, original)
    assert stop == original
    assert info is None


def test_executor_config_maps_optimization_fields() -> None:
    payload = {
        "mode": "paper",
        "symbol": "BTC/USDT:USDT",
        "timeframe": "15m",
        "informative_timeframe": "4h",
        "leverage": 20,
        "margin_mode": "cross",
        "max_open_positions": 1,
        "risk_per_trade": 0.035,
        "state_db_path": "state/test.db",
        "enable_vol_target_sizing": True,
        "vol_target_reference_pct": 0.3,
        "vol_target_min_scale": 0.2,
        "vol_target_max_scale": 2.5,
        "enable_atr_stop_bounds": True,
        "atr_stop_cap_multiple": 3.5,
        "atr_stop_floor_multiple": 1.5,
    }
    strategy = ExecutorConfig.from_dict(payload).to_scalp_strategy_config()
    assert strategy.enable_vol_target_sizing is True
    assert strategy.vol_target_reference_pct == 0.3
    assert strategy.vol_target_min_scale == 0.2
    assert strategy.vol_target_max_scale == 2.5
    assert strategy.enable_atr_stop_bounds is True
    assert strategy.atr_stop_cap_multiple == 3.5
    assert strategy.atr_stop_floor_multiple == 1.5


def test_gate_semantics_default_preserve_legacy_behaviour() -> None:
    strategy = StrategyConfig()
    assert strategy.shadow_daily_loss_pause_days == 0.0
    assert strategy.shadow_consecutive_loss_pause_days == 0.0
    assert strategy.shadow_drawdown_peak_reset is True
    assert strategy.shadow_halt_drawdown_pct == 0.0


def test_gate_semantics_extended_options_map_and_trigger() -> None:
    payload = {
        "mode": "live",
        "symbol": "BTC/USDT:USDT",
        "timeframe": "15m",
        "informative_timeframe": "4h",
        "leverage": 20,
        "margin_mode": "cross",
        "max_open_positions": 1,
        "risk_per_trade": 0.035,
        "state_db_path": "state/test.db",
        "enable_shadow_risk_gate": True,
        "shadow_consecutive_loss_stop": 3,
        "shadow_consecutive_loss_pause_days": 3.0,
        "shadow_drawdown_peak_reset": False,
        "shadow_halt_drawdown_pct": 25.0,
    }
    config = ExecutorConfig.from_dict(payload)
    assert config.shadow_consecutive_loss_pause_days == 3.0
    assert config.shadow_drawdown_peak_reset is False
    assert config.shadow_halt_drawdown_pct == 25.0

    engine = _make_engine(
        StrategyConfig(
            enable_shadow_risk_gate=True,
            shadow_consecutive_loss_stop=3,
            shadow_consecutive_loss_pause_days=3.0,
            shadow_drawdown_peak_reset=False,
            shadow_halt_drawdown_pct=25.0,
        )
    )
    # 连亏 3 笔 → 暂停到「平仓时刻 + 3 天」，而不是下一 UTC 日
    engine.capital = 1000.0
    exit_ts = 1_600_000_000.0
    for _ in range(3):
        engine._shadow_gate_after_close(-10.0, exit_ts)
    pause_until = float(engine.shadow_state.get("pause_until_ts", 0.0))
    assert abs(pause_until - (exit_ts + 3 * 86400.0)) < 1.0


def test_daily_alpha_min_dev_pct_default_is_off() -> None:
    assert StrategyConfig().daily_alpha_min_dev_pct is None


def test_daily_alpha_min_dev_pct_maps_through_executor() -> None:
    payload = {
        "mode": "paper",
        "symbol": "BTC/USDT:USDT",
        "timeframe": "15m",
        "informative_timeframe": "4h",
        "leverage": 20,
        "margin_mode": "cross",
        "max_open_positions": 1,
        "risk_per_trade": 0.035,
        "state_db_path": "state/test.db",
        "enable_daily_alpha_gate": True,
        "daily_alpha_max_dev_pct": 12.0,
        "daily_alpha_min_dev_pct": -5.0,
    }
    strategy = ExecutorConfig.from_dict(payload).to_scalp_strategy_config()
    assert strategy.daily_alpha_min_dev_pct == -5.0


def test_daily_alpha_gate_blocks_deep_discount_longs_only() -> None:
    """下限门：deep discount 禁多、允许空；cap 门：过热度禁多。"""
    engine = _make_engine(StrategyConfig(enable_daily_alpha_gate=True, daily_alpha_min_dev_pct=-5.0))
    # 直接替 precomputed 的日线乖离数组，验证门函数本身
    count = len(engine.mapping)
    engine.precomputed.daily_dev_pct_4h = [-8.0] * count
    assert engine._daily_alpha_gate_ok_for_idx(300, Direction.BULL) is False
    assert engine._daily_alpha_gate_ok_for_idx(300, Direction.BEAR) is True
    engine.precomputed.daily_dev_pct_4h = [-1.0] * count
    assert engine._daily_alpha_gate_ok_for_idx(300, Direction.BULL) is True
    engine.precomputed.daily_dev_pct_4h = [20.0] * count
    assert engine._daily_alpha_gate_ok_for_idx(300, Direction.BULL) is False


def test_daily_alpha_gate_floor_disabled_by_default() -> None:
    engine = _make_engine(StrategyConfig(enable_daily_alpha_gate=True))
    engine.precomputed.daily_dev_pct_4h = [-50.0] * len(engine.mapping)
    assert engine._daily_alpha_gate_ok_for_idx(300, Direction.BULL) is True
