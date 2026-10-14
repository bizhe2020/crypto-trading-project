"""scripts/audit_btc_scalp_sizing_cost.py 里纯函数的单元测试（合成输入，不依赖外部数据）。

覆盖：MFE 可达性表、近似 / 数值 Kelly（含两点分布的解析解）、对数增长、bootstrap 区间、
逐仓强平缓冲、手续费折算 R。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_btc_scalp_sizing_cost import (  # noqa: E402
    bootstrap_kelly,
    fee_r,
    kelly_fraction_approx,
    kelly_fraction_numeric,
    liquidation_buffer_pct,
    log_growth_per_trade,
    mfe_reach_table,
)


def test_mfe_reach_table_counts_levels_and_ignores_missing():
    table = mfe_reach_table([0.4, 1.0, 2.5, 5.0, None, float("nan")])
    assert table == {"0.5R": 75.0, "1R": 75.0, "2R": 50.0, "3R": 25.0, "4R": 25.0, "5R": 25.0}
    assert mfe_reach_table([]) == {"0.5R": 0.0, "1R": 0.0, "2R": 0.0, "3R": 0.0, "4R": 0.0, "5R": 0.0}


def test_kelly_numeric_matches_two_outcome_closed_form():
    # 赢 +2R / 输 -1R 各半：f* = p - (1-p)/b = 0.5 - 0.5/2 = 0.25
    values = [2.0, -1.0] * 50
    assert kelly_fraction_numeric(values) == pytest.approx(0.25, abs=1e-3)
    # 赢 +1R / 输 -1R、胜率 60%：f* = 0.6 - 0.4 = 0.2
    values = [1.0] * 60 + [-1.0] * 40
    assert kelly_fraction_numeric(values) == pytest.approx(0.2, abs=1e-3)


def test_kelly_numeric_zero_for_non_positive_edge_and_respects_ruin_bound():
    assert kelly_fraction_numeric([1.0, -1.0] * 10) == 0.0
    assert kelly_fraction_numeric([-0.5, -1.0]) == 0.0
    # 全是正收益时上界由 f_max 决定
    assert kelly_fraction_numeric([0.5, 1.0], f_max=0.3) == pytest.approx(0.3)


def test_kelly_approx_is_mean_over_variance():
    values = [2.0, -1.0] * 50
    mean = 0.5
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    assert kelly_fraction_approx(values) == pytest.approx(mean / var)
    assert kelly_fraction_approx([1.0, 1.0]) is None
    assert kelly_fraction_approx([1.0]) is None


def test_log_growth_per_trade_edges():
    values = [2.0, -1.0]
    assert log_growth_per_trade(values, 0.0) == 0.0
    expected = 0.5 * (math.log(1.5) + math.log(0.75))
    assert log_growth_per_trade(values, 0.25) == pytest.approx(expected)
    assert log_growth_per_trade(values, 1.0) == float("-inf")  # 1 + f * (-1) <= 0 -> 破产


def test_bootstrap_kelly_brackets_point_estimate():
    values = [2.0, -1.0] * 100
    lo, hi = bootstrap_kelly(values, n_boot=200)
    assert lo <= 0.25 <= hi
    assert lo >= 0.0


def test_liquidation_buffer_isolated_margin_approximation():
    assert liquidation_buffer_pct(1.2, 20) == pytest.approx(3.3)  # 5% - 0.5% - 1.2%
    assert liquidation_buffer_pct(1.2, 10) == pytest.approx(8.3)
    assert liquidation_buffer_pct(2.0, 20, maintenance_margin_pct=0.4) == pytest.approx(2.6)


def test_fee_r_uses_exchange_position_size():
    # 100 张 = 1 BTC；止损距离 500 USDT -> 风险 500；手续费 10 -> 0.02R
    assert fee_r(-10.0, 100, 50_000.0, 49_500.0) == pytest.approx(0.02)
    assert fee_r(-10.0, 0, 50_000.0, 49_500.0) is None
