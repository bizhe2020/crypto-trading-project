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
from scripts.client_paper_utils import ClientPaperState, build_client_paper_curve, load_or_init_state  # noqa: E402
from scripts.fetch_public_etf_history import fetch_timeframe, output_path_for  # noqa: E402
from scripts.market_schedule_utils import US_TZ, is_scheduled_window, is_us_trade_day, next_trade_day  # noqa: E402
from scripts.tqqq_cash_strict_utils import load_strict_frame_with_overlay_context, run_strict_candidate  # noqa: E402


DEFAULT_SIGNAL_CONFIG = ROOT / "config" / "tqqq_sqqq_signal_email.json"
DEFAULT_REPORT_DIR = ROOT / "var" / "reports"
LOCAL_TZ = ZoneInfo("Asia/Shanghai")


def frozen_tqqq_strict_config(signal_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_root": ROOT / str(signal_config.get("data_root", "data/public/etf")),
        "entry_fast_window": 25,
        "entry_slow_window": 150,
        "regime_filter": "ixic_filter",
        "max_hold_days": 90,
        "trailing_lookback_days": 10,
        "trailing_drawdown_pct": 12.0,
        "de_risk_signal_name": "breakout_fail_score_le3_flat",
        "recovery_reentry_rule": "score_ge3",
        "recovery_reentry_cooldown_days": 0,
        "switch_cost_bps": float(signal_config.get("switch_cost_bps", 10.0)),
        "initial_capital": 1000.0,
        "frozen_label": "tqqq_only_strict_recovery_frozen_20260526",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send TQQQ/SQQQ strategy signal email.")
    parser.add_argument("--mode", choices=["preopen", "close"], required=True)
    parser.add_argument("--signal-config", default=str(DEFAULT_SIGNAL_CONFIG))
    parser.add_argument("--recipient")
    parser.add_argument("--principal", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-send", action="store_true")
    return parser.parse_args()


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


def share_round_units(notional: float, price: float) -> int:
    if price <= 0 or notional <= 0:
        return 0
    return int(math.floor(notional / price))


def refresh_market_data(signal_config: dict[str, Any]) -> None:
    if not bool(signal_config.get("refresh_before_send", True)):
        return

    data_root = ROOT / str(signal_config.get("data_root", "data/public/etf"))
    timeframe = str(signal_config.get("timeframe", "1d"))
    proxy = str(signal_config.get("refresh_proxy") or "").strip()
    start = str(signal_config.get("refresh_start") or "2022-01-01T00:00:00Z")
    end = signal_config.get("refresh_end")
    sleep_seconds = float(signal_config.get("refresh_sleep_seconds", 0.25) or 0.25)
    symbols = ordered_unique([str(item) for item in signal_config.get("refresh_symbols", ["QQQ", "TQQQ", "SQQQ", "SPY", "^IXIC", "^VIX"])])

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 tqqq-sqqq-email-refresh"})
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
                raise RuntimeError(f"Refresh returned empty data for {symbol}")
            latest_ts = refreshed["date"].max()
            print(f"refreshed {symbol} {timeframe} -> {latest_ts.isoformat()} rows={len(refreshed)}")
            continue
        except Exception as exc:
            if proxy:
                print(f"refresh_proxy_failed_retry_direct {symbol} {timeframe}: {exc}")
                direct_session = requests.Session()
                direct_session.headers.update({"User-Agent": "Mozilla/5.0 tqqq-sqqq-email-refresh"})
                try:
                    refreshed = fetch_timeframe(
                        session=direct_session,
                        symbol=symbol,
                        timeframe=timeframe,
                        start=start,
                        end=end,
                        output_path=output_path,
                        sleep_seconds=sleep_seconds,
                        proxy=None,
                    )
                    if refreshed.empty:
                        raise RuntimeError(f"Refresh returned empty data for {symbol}")
                    latest_ts = refreshed["date"].max()
                    print(f"refreshed_direct {symbol} {timeframe} -> {latest_ts.isoformat()} rows={len(refreshed)}")
                    continue
                except Exception as retry_exc:
                    exc = retry_exc
            if bool(signal_config.get("refresh_fail_open", True)):
                print(f"refresh_failed_fallback {symbol} {timeframe}: {exc}")
                continue
            raise exc


def load_price_frame(data_root: Path, symbol: str) -> pd.DataFrame:
    frame = pd.read_feather(data_root / f"{symbol}-1d.feather")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["trade_day"] = frame["date"].dt.tz_convert(US_TZ).dt.date
    return frame[["trade_day", "open", "close"]].drop_duplicates("trade_day", keep="last").sort_values("trade_day").reset_index(drop=True)


def build_targets_by_day(path: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for _, row in path.iterrows():
        trade_day = str(pd.Timestamp(row["date"]).tz_convert(US_TZ).date())
        asset = str(row["position"])
        out[trade_day] = {
            "asset": asset,
            "leverage": 1.0 if asset in {"TQQQ", "SQQQ"} else 0.0,
            "signal_source_day": trade_day,
        }
    return out


def maybe_persist_anchor_fill(
    state_path: Path,
    state: ClientPaperState,
    *,
    asset: str,
    fill_price: float,
    units: int,
    persist_state: bool,
) -> ClientPaperState:
    if state.anchor_asset and state.anchor_fill_price > 0:
        return state
    updated = ClientPaperState(
        anchor_trade_day=state.anchor_trade_day,
        principal=state.principal,
        anchor_asset=asset,
        anchor_fill_price=float(fill_price),
        anchor_units=int(units),
    )
    if persist_state:
        write_json(
            state_path,
            {
                "anchor_trade_day": updated.anchor_trade_day,
                "principal": updated.principal,
                "anchor_asset": updated.anchor_asset,
                "anchor_fill_price": updated.anchor_fill_price,
                "anchor_units": updated.anchor_units,
            },
        )
    return updated


def build_payload(mode: str, signal_config: dict[str, Any], principal: float, *, persist_state: bool) -> dict[str, Any]:
    strict_cfg = frozen_tqqq_strict_config(signal_config)
    data_root = Path(strict_cfg["data_root"])
    frame = load_strict_frame_with_overlay_context(
        data_root=data_root,
        entry_fast_window=int(strict_cfg["entry_fast_window"]),
        entry_slow_window=int(strict_cfg["entry_slow_window"]),
    )
    result = run_strict_candidate(
        frame,
        regime_filter=str(strict_cfg["regime_filter"]),
        max_hold_days=int(strict_cfg["max_hold_days"]),
        trailing_lookback_days=int(strict_cfg["trailing_lookback_days"]),
        trailing_drawdown_pct=float(strict_cfg["trailing_drawdown_pct"]),
        switch_cost_bps=float(strict_cfg["switch_cost_bps"]),
        initial_capital=float(strict_cfg["initial_capital"]),
        de_risk_signal_name=str(strict_cfg.get("de_risk_signal_name", "off")),
        recovery_reentry_rule=str(strict_cfg.get("recovery_reentry_rule", "off")),
        recovery_reentry_cooldown_days=int(strict_cfg.get("recovery_reentry_cooldown_days", 0)),
    )
    path = result["path"].reset_index(drop=True)
    latest = path.iloc[-1]
    now_local = datetime.now(LOCAL_TZ)
    now_us = now_local.astimezone(US_TZ)
    action_day = now_us.date()
    subject_prefix = "开盘前" if mode == "preopen" else "收盘后"
    trade_day_label = "目标交易日" if mode == "preopen" else "收盘交易日"
    asset = str(latest["position"])
    reference_symbol = "TQQQ"
    price_frame = load_price_frame(data_root, reference_symbol)
    latest_price_row = price_frame.iloc[-1]
    reference_price = float(latest_price_row["close"])
    target_notional = principal if asset == "TQQQ" else 0.0
    suggested_units = share_round_units(target_notional, reference_price)
    actual_notional = suggested_units * reference_price
    cash_remaining = max(principal - actual_notional, 0.0)
    action_text = "做多 TQQQ" if asset == "TQQQ" else "空仓观望"

    state_path = ROOT / str(signal_config.get("client_paper_state_path", "state/tqqq_sqqq_client_paper_since_today.json"))
    state = load_or_init_state(state_path, action_day, principal, persist=persist_state)
    anchor_fill_price = float(signal_config.get("anchor_fill_price") or 0.0)
    anchor_units = int(signal_config.get("anchor_units") or 0)
    if state.anchor_trade_day == str(action_day) and asset in {"TQQQ", "SQQQ"} and anchor_fill_price > 0:
        state = maybe_persist_anchor_fill(
            state_path,
            state,
            asset=asset,
            fill_price=anchor_fill_price,
            units=anchor_units,
            persist_state=persist_state,
        )
    targets_by_day = build_targets_by_day(path)
    trade_days = price_frame[["trade_day"]].copy()
    paper_curve = build_client_paper_curve(
        trade_days=trade_days,
        price_by_asset={
            "TQQQ": load_price_frame(data_root, "TQQQ"),
        },
        targets_by_day=targets_by_day,
        anchor_trade_day=state.anchor_trade_day,
        principal=principal,
        anchor_fill_override={
            "asset": state.anchor_asset,
            "fill_price": state.anchor_fill_price,
            "units": state.anchor_units,
        },
    )
    if paper_curve.empty:
        paper_equity = principal
        paper_return_pct = 0.0
    else:
        paper_equity = float(paper_curve.iloc[-1]["equity_close"])
        paper_return_pct = round((paper_equity / principal - 1.0) * 100.0, 2) if principal > 0 else 0.0

    return {
        "mode": mode,
        "generated_at": now_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "subject_prefix": subject_prefix,
        "action_day": str(action_day),
        "trade_day_label": trade_day_label,
        "signal_source_date": str(pd.Timestamp(latest["date"]).tz_convert(US_TZ).date()),
        "position": asset,
        "action_text": action_text,
        "reference_price": round(reference_price, 4),
        "principal": round(principal, 2),
        "target_notional": round(target_notional, 2),
        "suggested_units": int(suggested_units),
        "actual_notional": round(actual_notional, 2),
        "cash_remaining": round(cash_remaining, 2),
        "client_paper_anchor_day": state.anchor_trade_day,
        "client_paper_equity": round(paper_equity, 2),
        "client_paper_return_pct": paper_return_pct,
        "anchor_fill_price": round(float(state.anchor_fill_price or 0.0), 4),
        "frozen_label": str(strict_cfg["frozen_label"]),
    }


def build_subject(payload: dict[str, Any]) -> str:
    action_tag = "做多TQQQ" if payload["position"] == "TQQQ" else "空仓"
    return f"[TQQQ][{action_tag}] {payload['subject_prefix']}策略信号 {payload['action_day']}"


def build_body(payload: dict[str, Any]) -> str:
    lines = [
        f"生成时间: {payload['generated_at']}",
        f"信号类型: {payload['subject_prefix']}信号",
        "执行标的: TQQQ",
        f"{payload['trade_day_label']}: {payload['action_day']}",
        "",
        f"当前策略动作: {payload['action_text']}",
        f"参考收盘价: {payload['reference_price']}",
        (f"实际买入价锚点: {payload['anchor_fill_price']}" if float(payload.get("anchor_fill_price") or 0.0) > 0 else ""),
        "",
        f"{payload['principal']:.2f} 美元本金下单参考:",
        f"目标名义仓位: {payload['target_notional']:.2f}",
        f"建议股数: {payload['suggested_units']}",
        f"对应名义仓位: {payload['actual_notional']:.2f}",
        f"现金剩余: {payload['cash_remaining']:.2f}",
    ]
    return "\n".join(lines)


def write_signal_files(mode: str, payload: dict[str, Any], subject: str, body: str) -> None:
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(DEFAULT_REPORT_DIR / f"tqqq_sqqq_email_signal_{mode}.json", payload)
    (DEFAULT_REPORT_DIR / f"tqqq_sqqq_email_signal_{mode}.txt").write_text(f"{subject}\n\n{body}\n")


def main() -> None:
    args = parse_args()
    signal_config = load_json(Path(args.signal_config))
    recipient = str(args.recipient or signal_config.get("recipient") or "").strip()
    principal = float(args.principal if args.principal is not None else signal_config.get("principal", 5000.0))
    if not recipient:
        raise SystemExit("Missing recipient.")

    if not args.force_send:
        now_local = datetime.now(LOCAL_TZ)
        now_us = now_local.astimezone(US_TZ)
        if not is_us_trade_day(now_us.date()):
            print("skip_non_trade_day", now_us.date())
            return
        if not is_scheduled_window(str(args.mode), "us", now_local):
            print("skip_outside_window", now_local.isoformat())
            return

    refresh_market_data(signal_config)
    payload = build_payload(str(args.mode), signal_config, principal, persist_state=not args.dry_run)
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
