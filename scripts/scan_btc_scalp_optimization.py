#!/usr/bin/env python3
"""BTC scalp 优化扫描（唯一无前视口径）。

在权威数据上、用 `scripts/live_readiness_report.load_prepared_data` 的唯一无前视
口径加载一次数据，然后对若干候选配置逐一回测，输出跨窗口对比表。

**口径约束**：本脚本不新增任何前视口径开关；所有回测都走
`build_precomputed_state_confirmed_4h`（见 docs/backtest_lookahead_policy.md）。

用法::

    python3 scripts/scan_btc_scalp_optimization.py --set risk
    python3 scripts/scan_btc_scalp_optimization.py --set exit --windows 2022-01-01,2025-01-01
    python3 scripts/scan_btc_scalp_optimization.py --set risk,exit --output var/tmp/btc_opt/scan.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.live_readiness_report import (  # noqa: E402
    compact_metrics,
    load_prepared_data,
    run_engine,
    trade_dataframe,
)

DEFAULT_CONFIG = ROOT / "var" / "tokyo_audit" / "btc_scalp_20260920" / "config.live.high-leverage-structure.json"
DEFAULT_DATA_15M = ROOT / "var" / "tokyo_audit" / "btc_scalp_20260920" / "BTC_USDT_USDT-15m-futures.feather"
DEFAULT_DATA_4H = ROOT / "var" / "tokyo_audit" / "btc_scalp_20260920" / "BTC_USDT_USDT-4h-futures.feather"
DEFAULT_OUTPUT = ROOT / "var" / "tmp" / "btc_opt" / "scan_result.json"

# 每个候选 = 基线配置上的覆盖键。基线永远是同一份 live 配置。
VARIANT_SETS: dict[str, dict[str, dict[str, Any]]] = {
    # ---- 风险预算 / 仓位（不需要新 edge 就能改善的结果） ----
    "risk": {
        "risk_eq_0035": {
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.035,
        },
        "risk_eq_0020": {
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.02,
        },
        "risk_cap_all_0035": {
            "enable_regime_directional_risk": True,
            "bull_strong_long_risk_per_trade": 0.035,
            "bull_strong_short_risk_per_trade": 0.035,
            "bull_weak_long_risk_per_trade": 0.035,
            "bull_weak_short_risk_per_trade": 0.035,
            "bear_weak_long_risk_per_trade": 0.035,
            "bear_weak_short_risk_per_trade": 0.035,
            "bear_strong_long_risk_per_trade": 0.035,
            "bear_strong_short_risk_per_trade": 0.035,
        },
        "risk_cap_bull_strong_0035": {
            "enable_regime_directional_risk": True,
            "bull_strong_long_risk_per_trade": 0.035,
        },
    },
    # ---- 出场几何 ----
    "exit": {
        "exit_no_fixed_target": {"disable_fixed_target_exit": True},
        "exit_pressure_off": {"enable_pressure_level_trailing": False},
        "exit_atr_trail_off": {"enable_atr_trailing": False},
        "exit_target_cap_low": {
            "enable_target_rr_cap": True,
            "loose_target_rr_cap": 3.0,
            "normal_target_rr_cap": 2.5,
            "tight_target_rr_cap": 2.0,
        },
        "exit_loose_trail_1r_05": {
            "loose_stage0_trigger_r": 1.0,
            "loose_stage0_lock_r": 0.5,
            "loose_stage1_trigger_r": 2.0,
            "loose_stage1_lock_r": 1.0,
        },
        "exit_no_pressure_no_fixed": {
            "enable_pressure_level_trailing": False,
            "disable_fixed_target_exit": True,
        },
    },
    # ---- 最终候选（全窗口 2022 起：风险归一 + 关 ATR 追踪 + vol target） ----
    "final": {
        "f_risk35_atr_off_vol030": {
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.035,
            "enable_atr_trailing": False,
            "enable_vol_target_sizing": True,
            "vol_target_reference_pct": 0.30,
        },
        "f_risk35_atr_off_vol030_cap1": {
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.035,
            "enable_atr_trailing": False,
            "enable_vol_target_sizing": True,
            "vol_target_reference_pct": 0.30,
            "vol_target_max_scale": 1.0,
        },
        "f_risk35_vol030_cap1": {
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.035,
            "enable_vol_target_sizing": True,
            "vol_target_reference_pct": 0.30,
            "vol_target_max_scale": 1.0,
        },
        "f_risk35_atr_off_vol045_cap1": {
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.035,
            "enable_atr_trailing": False,
            "enable_vol_target_sizing": True,
            "vol_target_reference_pct": 0.45,
            "vol_target_max_scale": 1.0,
        },
        "f_risk35_atr_off": {
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.035,
            "enable_atr_trailing": False,
        },
    },
    # ---- 组合候选（在 2025+ 窗口上单项最优的叠加） ----
    "core": {
        "c_risk35": {
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.035,
        },
        "c_risk35_atr_off": {
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.035,
            "enable_atr_trailing": False,
        },
        "c_risk35_atr_off_vol030": {
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.035,
            "enable_atr_trailing": False,
            "enable_vol_target_sizing": True,
            "vol_target_reference_pct": 0.30,
        },
        "c_risk35_atr_off_vol030_atrcap35": {
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.035,
            "enable_atr_trailing": False,
            "enable_vol_target_sizing": True,
            "vol_target_reference_pct": 0.30,
            "enable_atr_stop_bounds": True,
            "atr_stop_cap_multiple": 3.5,
        },
        "c_risk35_atr_off_nopressure_nofixed": {
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.035,
            "enable_atr_trailing": False,
            "enable_pressure_level_trailing": False,
            "disable_fixed_target_exit": True,
        },
        "c_vol030_only": {
            "enable_vol_target_sizing": True,
            "vol_target_reference_pct": 0.30,
        },
        "c_atr_off_only": {"enable_atr_trailing": False},
        "c_vol030_atr_off": {
            "enable_vol_target_sizing": True,
            "vol_target_reference_pct": 0.30,
            "enable_atr_trailing": False,
        },
    },
    # ---- 波动率目标仓位（15m ATR% 中位约 0.31%，见 tools 说明） ----
    "vol": {
        "vol_target_ref_030": {
            "enable_vol_target_sizing": True,
            "vol_target_reference_pct": 0.30,
        },
        "vol_target_ref_045": {
            "enable_vol_target_sizing": True,
            "vol_target_reference_pct": 0.45,
        },
        "vol_target_ref_020": {
            "enable_vol_target_sizing": True,
            "vol_target_reference_pct": 0.20,
        },
    },
    # ---- ATR 止损宽度钳制 ----
    "atrstop": {
        "atr_cap_35": {
            "enable_atr_stop_bounds": True,
            "atr_stop_cap_multiple": 3.5,
        },
        "atr_cap_25": {
            "enable_atr_stop_bounds": True,
            "atr_stop_cap_multiple": 2.5,
        },
        "atr_floor_20": {
            "enable_atr_stop_bounds": True,
            "atr_stop_floor_multiple": 2.0,
        },
        "atr_band_20_35": {
            "enable_atr_stop_bounds": True,
            "atr_stop_floor_multiple": 2.0,
            "atr_stop_cap_multiple": 3.5,
        },
    },
    # ---- ATR 追踪止损参数（当前配置激活 2.06R、倍数 2.7/2.25/1.8） ----
    "atrtrail": {
        "atr_act_30": {"atr_activation_rr": 3.0},
        "atr_act_40": {"atr_activation_rr": 4.0},
        "atr_loose_mult": {
            "atr_loose_multiplier": 3.5,
            "atr_normal_multiplier": 3.0,
            "atr_tight_multiplier": 2.5,
        },
        "atr_act_30_loose_mult": {
            "atr_activation_rr": 3.0,
            "atr_loose_multiplier": 3.5,
            "atr_normal_multiplier": 3.0,
            "atr_tight_multiplier": 2.5,
        },
    },
    # ---- 日线乖离 alpha 门 / regime 过滤（当前 200 日 SMA、12% 乖离） ----
    "regime": {
        "alpha_sma_100": {"daily_alpha_sma_period": 100},
        "alpha_sma_300": {"daily_alpha_sma_period": 300},
        "dir_regime_switch_off": {"enable_directional_regime_switch": False},
    },
    # ---- 日线乖离「下限」门（形态级归因发现：深贴水区多头期望≈0） ----
    # 见 docs/btc_scalp_optimization_20260921.md §11；阈值 -3.6 来自 max-statistic 置换检验
    # （p=0.075，六特征中最好但仍未过 0.05，故必须做引擎级 + 季度切片验证）。
    "devfloor": {
        "df_m36": {"daily_alpha_min_dev_pct": -3.6},
        "df_m50": {"daily_alpha_min_dev_pct": -5.0},
        "df_m80": {"daily_alpha_min_dev_pct": -8.0},
        "df_zero": {"daily_alpha_min_dev_pct": 0.0},
        "df_m36_risk35_vol": {
            "daily_alpha_min_dev_pct": -3.6,
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.035,
            "enable_vol_target_sizing": True,
            "vol_target_reference_pct": 0.30,
            "vol_target_max_scale": 1.0,
        },
    },
    # ---- 季度切片用：2022 起窗口核对 ATR 追踪层与风险归一化 ----
    "atrq": {
        "q_risk35": {
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.035,
        },
        "q_risk35_atr_off": {
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.035,
            "enable_atr_trailing": False,
        },
        "q_atr_off_only": {"enable_atr_trailing": False},
        "q_risk35_atr_off_vol030_cap1": {
            "enable_regime_directional_risk": False,
            "risk_per_trade": 0.035,
            "enable_atr_trailing": False,
            "enable_vol_target_sizing": True,
            "vol_target_reference_pct": 0.30,
            "vol_target_max_scale": 1.0,
        },
    },
    # ---- 引擎级 shadow gate A/B（必须显式打开 enable_shadow_risk_gate_backtest） ----
    # memory 明确警告：动态杠杆 + shadow gate + regime 切换导致强路径依赖，
    # 闸门必须引擎级 A/B，不能用「固定交易序列 + 后台过滤」的线性推理。
    "enginegate": {
        "eg_current": {
            "enable_shadow_risk_gate": True,
            "enable_shadow_risk_gate_backtest": True,
        },
        "eg_dd12_cool7_no_reset": {
            "enable_shadow_risk_gate": True,
            "enable_shadow_risk_gate_backtest": True,
            "shadow_equity_drawdown_stop_pct": 12.0,
            "shadow_equity_drawdown_cooldown_days": 7,
            "shadow_drawdown_peak_reset": False,
            "shadow_daily_loss_pause_days": 1.0,
        },
        "eg_dd10_cool7_no_reset": {
            "enable_shadow_risk_gate": True,
            "enable_shadow_risk_gate_backtest": True,
            "shadow_equity_drawdown_stop_pct": 10.0,
            "shadow_equity_drawdown_cooldown_days": 7,
            "shadow_drawdown_peak_reset": False,
            "shadow_daily_loss_pause_days": 1.0,
        },
        "eg_dd8_cool7_no_reset": {
            "enable_shadow_risk_gate": True,
            "enable_shadow_risk_gate_backtest": True,
            "shadow_equity_drawdown_stop_pct": 8.0,
            "shadow_equity_drawdown_cooldown_days": 7,
            "shadow_drawdown_peak_reset": False,
            "shadow_daily_loss_pause_days": 1.0,
        },
        "eg_dd10_cool7_no_reset_streak2": {
            "enable_shadow_risk_gate": True,
            "enable_shadow_risk_gate_backtest": True,
            "shadow_equity_drawdown_stop_pct": 10.0,
            "shadow_equity_drawdown_cooldown_days": 7,
            "shadow_drawdown_peak_reset": False,
            "shadow_daily_loss_pause_days": 1.0,
            "shadow_consecutive_loss_stop": 3,
            "shadow_consecutive_loss_pause_days": 2.0,
        },
    },
    # ---- 与 memory/daily_overextension_gate_20260909.md 的无前视口径对齐 ----
    # 期望（2023-01-01 起，initial_capital=1000，confirmed_4h）：
    #   off=393 / cap8=17256 / cap10=26192 / cap12=45184 / cap14=27578 / cap16=13877
    "alphagate": {
        "alpha_off": {"enable_daily_alpha_gate": False},
        "alpha_cap8": {"daily_alpha_max_dev_pct": 8.0},
        "alpha_cap10": {"daily_alpha_max_dev_pct": 10.0},
        "alpha_cap14": {"daily_alpha_max_dev_pct": 14.0},
        "alpha_cap16": {"daily_alpha_max_dev_pct": 16.0},
    },
    # ---- 入场过滤（引擎层可表达的） ----
    "filter": {
        "filter_alpha_gate_8pct": {"daily_alpha_max_dev_pct": 8.0},
        "filter_alpha_gate_off": {"enable_daily_alpha_gate": False},
        "filter_hfvf_off": {"use_hfvf_filter": False},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BTC scalp 优化扫描（无前视口径）")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-15m", default=str(DEFAULT_DATA_15M))
    parser.add_argument("--data-4h", default=str(DEFAULT_DATA_4H))
    parser.add_argument("--set", default="risk", help="逗号分隔的候选集：risk/exit/filter/all")
    parser.add_argument("--windows", default="2022-01-01,2025-01-01", help="逗号分隔的回测起始日")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--include-base", action="store_true", default=True)
    return parser.parse_args()


def load_base_config(path: Path) -> dict[str, Any]:
    """读取配置并剔除只在实盘执行层生效、与回测无关的密钥字段。"""
    payload = json.loads(path.read_text())
    for key in ("api_key", "api_secret", "api_passphrase", "telegram_token", "telegram_chat_id"):
        payload.pop(key, None)
    return payload


def trade_stats(engine: Any) -> dict[str, Any]:
    trades = trade_dataframe(engine)
    if trades.empty:
        return {
            "net_r_sum": 0.0,
            "avg_net_r": 0.0,
            "median_net_r": 0.0,
            "gross_r_sum": 0.0,
            "avg_fee_r": 0.0,
            "avg_bars_held": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }
    # Trade dataclass 没有 risk_amount 字段，用「数量 × 初始止损距离」重建。
    risk = (trades["quantity"] * (trades["entry_price"] - trades["initial_stop_price"]).abs()).replace(
        0.0, float("nan")
    )
    net_r = trades["pnl"] / risk
    gross_r = trades["gross_pnl"] / risk
    fee_r = trades["fees"] / risk
    return {
        "net_r_sum": round(float(net_r.sum()), 3),
        "avg_net_r": round(float(net_r.mean()), 4),
        "median_net_r": round(float(net_r.median()), 4),
        "gross_r_sum": round(float(gross_r.sum()), 3),
        "avg_fee_r": round(float(fee_r.mean()), 4),
        "avg_bars_held": round(float((trades["exit_idx"] - trades["entry_idx"]).mean()), 1),
        "long_trades": int((trades["direction"] == "BULL").sum()),
        "short_trades": int((trades["direction"] == "BEAR").sum()),
    }


def run_variant(
    name: str,
    payload: dict[str, Any],
    prepared: Any,
    windows: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "windows": {}}
    for window in windows:
        started = time.time()
        metrics, engine = run_engine(payload, prepared, window)
        compact = compact_metrics(metrics)
        compact.update(trade_stats(engine))
        compact["initial_capital"] = round(float(metrics.get("initial_capital", 0.0)), 2)
        compact["final_capital"] = round(float(metrics.get("final_capital", 0.0)), 2)
        compact["elapsed_s"] = round(time.time() - started, 1)
        result["windows"][window] = compact
    return result


def print_table(results: list[dict[str, Any]], windows: list[str]) -> None:
    header = f"{'variant':<32}" + "".join(f"{w:>34}" for w in windows)
    print(header)
    print("-" * len(header))
    for item in results:
        row = f"{item['name']:<32}"
        for window in windows:
            m = item["windows"].get(window, {})
            cell = (
                f"{m.get('total_return_pct', 0):>9.1f}% "
                f"dd{m.get('max_drawdown_pct', 0):>5.1f} "
                f"sh{m.get('sharpe_ratio', 0):>5.2f} "
                f"n{m.get('total_trades', 0):>4d}"
            )
            row += f"{cell:>34}"
        print(row)


def main() -> None:
    args = parse_args()
    sets: list[str] = []
    for token in str(args.set).split(","):
        token = token.strip()
        if not token:
            continue
        if token == "all":
            sets = list(VARIANT_SETS)
            break
        if token not in VARIANT_SETS:
            raise SystemExit(f"未知候选集: {token}（可选: {', '.join(VARIANT_SETS)}, all）")
        sets.append(token)
    windows = [w.strip() for w in str(args.windows).split(",") if w.strip()]

    base_payload = load_base_config(Path(args.config))
    start = pd.Timestamp(min(windows), tz="UTC")
    print(f"加载数据 start>={start} ...", flush=True)
    t0 = time.time()
    prepared = load_prepared_data(
        Path(args.data_15m),
        Path(args.data_4h),
        start,
        base_payload.get("regime_switcher_thresholds"),
    )
    print(
        f"数据加载完成 {time.time() - t0:.1f}s  15m={len(prepared.c15m)} 4h={len(prepared.c4h)} "
        f"范围={prepared.start} → {prepared.end}",
        flush=True,
    )

    variants: list[tuple[str, dict[str, Any]]] = []
    if args.include_base:
        variants.append(("base(live)", dict(base_payload)))
    for set_name in sets:
        for name, overrides in VARIANT_SETS[set_name].items():
            payload = dict(base_payload)
            payload.update(overrides)
            variants.append((f"{set_name}.{name}", payload))

    results: list[dict[str, Any]] = []
    for name, payload in variants:
        print(f"→ 回测 {name} ...", flush=True)
        results.append(run_variant(name, payload, prepared, windows))

    print()
    print_table(results, windows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "config": str(Path(args.config).resolve()),
                "data_15m": str(Path(args.data_15m).resolve()),
                "data_4h": str(Path(args.data_4h).resolve()),
                "windows": windows,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"\n输出: {output_path}")


if __name__ == "__main__":
    main()
