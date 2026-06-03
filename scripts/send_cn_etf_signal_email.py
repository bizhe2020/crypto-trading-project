#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.client_email_utils import load_json, send_email, write_json  # noqa: E402
from scripts.client_paper_utils import build_client_paper_curve, load_or_init_state  # noqa: E402
from scripts.cn_nasdaq100_strict_utils import load_config as load_strict_config, load_strict_frame, run_strict_path  # noqa: E402
from scripts.fetch_public_etf_history import fetch_timeframe, output_path_for  # noqa: E402
from scripts.market_schedule_utils import is_cn_trade_day, is_scheduled_window  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "config.paper.cn-nasdaq100-etf.json"
DEFAULT_SIGNAL_CONFIG = ROOT / "config" / "cn_etf_signal_email.json"
DEFAULT_REPORT_DIR = ROOT / "var" / "reports"
LOCAL_TZ = ZoneInfo("Asia/Shanghai")


def strict_frozen_strategy_config(path: Path) -> dict[str, Any]:
    config = load_strict_config(path)
    config["conditional_leverage_enabled"] = False
    config["conditional_leverage_value"] = 1.0
    config["tiered_leverage_enabled"] = True
    config["tiered_leverage_rules"] = [
        {
            "vix_label": "vix_normal",
            "rel_strength_label": "qqq_strong",
            "leverage": 2.0,
        },
        {
            "vix_label": "vix_normal",
            "rel_strength_label": "qqq_neutral",
            "leverage": 1.5,
        },
    ]
    config["entry_fast_window"] = 21
    config["entry_slow_window"] = 200
    config["regime_filter"] = "ixic_filter"
    config["max_hold_days"] = 120
    config["trailing_lookback_days"] = 4
    config["trailing_drawdown_pct"] = 4.0
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send CN ETF 513100 strategy signal email.")
    parser.add_argument("--mode", choices=["preopen", "close"], required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--signal-config", default=str(DEFAULT_SIGNAL_CONFIG))
    parser.add_argument("--recipient")
    parser.add_argument("--principal", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-send", action="store_true")
    return parser.parse_args()


def lot_round_shares(notional: float, price: float, lot_size: int = 100) -> int:
    if price <= 0 or notional <= 0:
        return 0
    raw = int(math.floor(notional / price))
    return (raw // lot_size) * lot_size


def ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalized = str(item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def refresh_market_data(strategy_config: dict[str, Any], signal_config: dict[str, Any]) -> None:
    if not bool(signal_config.get("refresh_before_send", True)):
        return
    timeframe = str(strategy_config.get("timeframe", "1d"))
    data_root = ROOT / str(strategy_config.get("data_root", "data/public/etf"))
    proxy = str(signal_config.get("refresh_proxy") or "").strip()
    start = str(signal_config.get("refresh_start") or "2022-01-01T00:00:00Z")
    end = signal_config.get("refresh_end")
    sleep_seconds = float(signal_config.get("refresh_sleep_seconds", 0.25) or 0.25)

    explicit_symbols = signal_config.get("refresh_symbols")
    if isinstance(explicit_symbols, list) and explicit_symbols:
        symbols = ordered_unique([str(symbol) for symbol in explicit_symbols])
    else:
        regime_symbols = strategy_config.get("regime_symbols", {})
        symbols = ordered_unique(
            [
                str(strategy_config.get("signal_symbol", "QQQ")),
                str(strategy_config.get("execution_symbol", "513100.SS")),
                str(regime_symbols.get("spy", "SPY")),
                str(regime_symbols.get("ixic", "^IXIC")),
                str(regime_symbols.get("vix", "^VIX")),
            ]
        )

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 cn-etf-email-refresh"})
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
        print(f"refresh_proxy {proxy}")

    for symbol in symbols:
        output_path = output_path_for(data_root, symbol, timeframe)
        try:
            refreshed = fetch_timeframe(
                session=session,
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                output_path=output_path,
                sleep_seconds=sleep_seconds,
                proxy=proxy,
            )
            if refreshed.empty:
                raise RuntimeError(f"Refresh returned empty data for {symbol} {timeframe}")
            latest_ts = refreshed["date"].max()
            print(f"refreshed {symbol} {timeframe} -> {latest_ts.isoformat()} rows={len(refreshed)}")
        except Exception as exc:
            if bool(signal_config.get("refresh_fail_open", True)):
                print(f"refresh_failed_fallback {symbol} {timeframe}: {exc}")
                continue
            raise


def build_targets_by_day(path: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if path.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in path.iterrows():
        trade_day = str(pd.Timestamp(row["date"]).date())
        asset = "513100.SS" if str(row["position"]) == "LONG" else "CASH"
        leverage = float(row.get("leverage", 0.0) or 0.0) if asset != "CASH" else 0.0
        out[trade_day] = {
            "asset": asset,
            "leverage": leverage,
            "signal_source_day": trade_day,
        }
    return out


def load_execution_prices(strategy_config: dict[str, Any]) -> pd.DataFrame:
    data_root = ROOT / str(strategy_config.get("data_root", "data/public/etf"))
    execution_symbol = str(strategy_config.get("execution_symbol", "513100.SS"))
    frame = pd.read_feather(data_root / f"{execution_symbol}-1d.feather")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["trade_day"] = frame["date"].dt.tz_convert(LOCAL_TZ).dt.date
    return frame[["trade_day", "open", "close"]].drop_duplicates("trade_day", keep="last").sort_values("trade_day").reset_index(drop=True)


def build_payload(
    mode: str,
    config_path: Path,
    strategy_config: dict[str, Any],
    signal_config: dict[str, Any],
    principal: float,
    *,
    persist_state: bool,
) -> dict[str, Any]:
    strict_config = strict_frozen_strategy_config(config_path)
    frame = load_strict_frame(strict_config)
    path = run_strict_path(frame, strict_config).reset_index(drop=True)
    now_local = datetime.now(LOCAL_TZ)
    action_day = now_local.date()
    subject_prefix = "开盘前" if mode == "preopen" else "收盘后"
    reference_price_label = "上一交易日收盘参考价" if mode == "preopen" else "今日收盘价"
    trade_day_label = "目标交易日" if mode == "preopen" else "收盘交易日"

    if path.empty:
        raise RuntimeError("Strict frozen CN ETF path is empty.")
    target_row = path[path["date"] == pd.Timestamp(action_day)]
    if target_row.empty:
        target_row = path[path["date"] <= pd.Timestamp(action_day)]
    if target_row.empty:
        raise RuntimeError("No strict frozen CN ETF signal available for current action day.")
    latest = target_row.iloc[-1]
    signal_source_date = str(latest.get("signal_us_day") or "")

    position = str(latest["position"])
    leverage = float(latest.get("leverage", 0.0) or 0.0)
    execution_symbol = str(strategy_config.get("execution_symbol", "513100.SS"))
    reference_price = float(latest.get("asset_open", 0.0) or 0.0) if mode == "preopen" else float(latest.get("asset_close", 0.0) or 0.0)
    target_notional = principal * leverage if position == "LONG" else 0.0
    suggested_shares = lot_round_shares(target_notional, reference_price, 100)
    actual_notional = suggested_shares * reference_price
    financing_needed = max(actual_notional - principal, 0.0)
    cash_remaining = max(principal - min(actual_notional, principal), 0.0)
    action_text = f"做多 {execution_symbol}" if position == "LONG" else "空仓观望"

    state_path = ROOT / str(signal_config.get("client_paper_state_path", "state/cn_etf_client_paper_since_today.json"))
    state = load_or_init_state(state_path, action_day, principal, persist=persist_state)
    execution_prices = load_execution_prices(strategy_config)
    targets_by_day = build_targets_by_day(path)
    paper_curve = build_client_paper_curve(
        trade_days=execution_prices[["trade_day"]].copy(),
        price_by_asset={"513100.SS": execution_prices},
        targets_by_day=targets_by_day,
        anchor_trade_day=state.anchor_trade_day,
        principal=principal,
    )
    if paper_curve.empty:
        paper_equity = principal
        paper_return_pct = 0.0
    else:
        paper_equity = float(paper_curve.iloc[-1]["equity_close"])
        paper_return_pct = round((paper_equity / principal - 1.0) * 100.0, 2) if principal > 0 else 0.0

    payload = {
        "mode": mode,
        "generated_at": now_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "subject_prefix": subject_prefix,
        "action_day": str(action_day),
        "trade_day_label": trade_day_label,
        "signal_source_date": signal_source_date,
        "execution_symbol": execution_symbol,
        "position": position,
        "action_text": action_text,
        "reference_price_label": reference_price_label,
        "reference_price": round(reference_price, 4),
        "leverage": round(leverage, 4),
        "principal": round(principal, 2),
        "target_notional": round(target_notional, 2),
        "suggested_shares": int(suggested_shares),
        "actual_notional": round(actual_notional, 2),
        "financing_needed": round(financing_needed, 2),
        "cash_remaining": round(cash_remaining, 2),
        "data_quality": {
            "incomplete_rows": 0,
            "latest_complete_date": str(pd.Timestamp(latest["date"]).date()),
        },
        "client_paper_anchor_day": state.anchor_trade_day,
        "client_paper_equity": round(paper_equity, 2),
        "client_paper_return_pct": paper_return_pct,
        "frozen_label": "strict_cn_nasdaq100_frozen_20260523",
    }
    return payload


def build_subject(payload: dict[str, Any]) -> str:
    action_tag = "做多" if payload["position"] == "LONG" else "空仓"
    return f"[513100][{action_tag}] {payload['subject_prefix']}策略信号 {payload['action_day']}"


def build_body(payload: dict[str, Any]) -> str:
    lines = [
        f"生成时间: {payload['generated_at']}",
        f"信号类型: {payload['subject_prefix']}信号",
        "执行标的: 国泰纳指100ETF(513100)",
        f"{payload['trade_day_label']}: {payload['action_day']}",
    ]
    data_quality = payload.get("data_quality", {}) or {}
    if int(data_quality.get("incomplete_rows", 0) or 0) > 0:
        lines.extend(
            [
                "数据状态: 存在缺失数据，已自动跳过不完整交易日",
                f"信号依据日期: {data_quality.get('latest_complete_date') or payload['signal_source_date']}",
            ]
        )
    lines.extend(
        [
            "",
            f"当前策略动作: {payload['action_text']}",
            f"目标杠杆: {payload['leverage']}x",
            f"{payload['reference_price_label']}: {payload['reference_price']}",
            "",
            "10000本金下单参考:" if abs(payload["principal"] - 10000.0) < 1e-9 else f"{payload['principal']:.2f} 本金下单参考:",
            f"目标名义仓位: {payload['target_notional']:.2f}",
            f"建议份额(按100份整手): {payload['suggested_shares']}",
            f"对应名义仓位: {payload['actual_notional']:.2f}",
            f"预计融资占用: {payload['financing_needed']:.2f}",
            f"自有现金剩余: {payload['cash_remaining']:.2f}",
        ]
    )
    return "\n".join(lines)


def write_signal_files(mode: str, payload: dict[str, Any], subject: str, body: str) -> tuple[Path, Path]:
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DEFAULT_REPORT_DIR / f"cn_etf_513100_email_signal_{mode}.json"
    txt_path = DEFAULT_REPORT_DIR / f"cn_etf_513100_email_signal_{mode}.txt"
    write_json(json_path, payload)
    txt_path.write_text(f"{subject}\n\n{body}\n")
    return json_path, txt_path


def main() -> None:
    args = parse_args()
    strategy_config = strict_frozen_strategy_config(Path(args.config))
    signal_config = load_json(Path(args.signal_config))
    recipient = str(args.recipient or signal_config.get("recipient") or "").strip()
    principal = float(args.principal if args.principal is not None else signal_config.get("principal", 10000.0))
    if not recipient:
        raise SystemExit("Missing recipient.")

    if not args.force_send:
        now_local = datetime.now(LOCAL_TZ)
        if not is_cn_trade_day(now_local.date()):
            print("skip_non_trade_day", now_local.date())
            return
        if not is_scheduled_window(str(args.mode), "cn", now_local):
            print("skip_outside_window", now_local.isoformat())
            return

    refresh_market_data(strategy_config, signal_config)
    payload = build_payload(str(args.mode), Path(args.config), strategy_config, signal_config, principal, persist_state=not args.dry_run)
    subject = build_subject(payload)
    body = build_body(payload)
    write_signal_files(str(args.mode), payload, subject, body)

    if args.dry_run:
        print(subject)
        print()
        print(body)
        return

    send_email(signal_config, recipient, subject, body)
    print(subject)
    print("sent_to", recipient)


if __name__ == "__main__":
    main()
