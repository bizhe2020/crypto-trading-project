"""趋势投票篮子腿：用 BTC scalp 的闲置保证金，在独立合约（XAU / QQQ 永续）上跑日线趋势投票。

设计要点（与 research/time_in_market/explore.py 的 basket2_gld_qqq 口径一致）：
  * 信号：ETF 日线（GLD / QQQ，Yahoo）的 5 规则投票 strategy.trend_vote.vote_target，只用已收盘日线。
  * 执行：每个新的已收盘交易日，把各腿多头名义调到 k × 账户权益 × 腿权重 × 投票比例。
  * 只做多、逐仓、市价单；真实仓位以交易所为准（每次都从交易所读取，本地只存审计和去重）。
  * 与 scalp 隔离：不碰 scalp 的 BTC-USDT-SWAP；篮子保证金上限 max_basket_margin_frac × 权益，
    给 scalp 留出保证金。
  * dry_run=true 时只读账户并记录计划单，不下单。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.okx_client import OkxClient, OkxCredentials  # noqa: E402
from bot.state_store import StateStore  # noqa: E402
from strategy.trend_vote import VOTE_RULES, vote_components, vote_target  # noqa: E402

NEW_YORK = ZoneInfo("America/New_York")
STATE_KEY = "trend_basket_state"


@dataclass
class BasketLeg:
    name: str
    signal_symbol: str  # Yahoo 代码，如 GLD / QQQ
    execution_symbol: str  # ccxt 统一代码，如 XAU/USDT:USDT
    weight: float = 0.5


@dataclass
class TrendBasketConfig:
    mode: str = "live"
    dry_run: bool = True
    enabled: bool = True
    # 复用 scalp 实盘配置里的 api_key/api_secret/api_passphrase/proxy，避免再复制一份密钥。
    credentials_config_path: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    api_passphrase: str | None = None
    proxy: str | None = None
    state_db_path: str = "state/trend_basket_live.db"
    signal_data_dir: str = "var/trend_basket/etf"
    legs: list[BasketLeg] = field(default_factory=list)
    basket_multiplier: float = 1.0
    leverage: int = 6
    margin_mode: str = "isolated"
    rebalance_band_frac: float = 0.1
    max_basket_margin_frac: float = 0.2
    min_order_notional_usdt: float = 20.0
    ny_close_buffer_minutes: int = 15
    history_days: int = 500
    poll_interval_seconds: int = 300
    failure_retry_seconds: int = 900

    @classmethod
    def from_dict(cls, payload: dict[str, Any], base_dir: Path | None = None) -> "TrendBasketConfig":
        data = dict(payload)
        legs = [BasketLeg(**leg) for leg in data.pop("legs", [])]
        known = {f for f in cls.__dataclass_fields__}
        cfg = cls(**{k: v for k, v in data.items() if k in known}, legs=legs)
        cred_path = cfg.credentials_config_path
        if cred_path and not (cfg.api_key and cfg.api_secret and cfg.api_passphrase):
            path = Path(cred_path)
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            creds = json.loads(path.read_text())
            cfg.api_key = cfg.api_key or creds.get("api_key")
            cfg.api_secret = cfg.api_secret or creds.get("api_secret")
            cfg.api_passphrase = cfg.api_passphrase or creds.get("api_passphrase")
            if cfg.proxy is None:
                cfg.proxy = creds.get("proxy")
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.legs:
            raise ValueError("trend basket needs at least one leg")
        names = [leg.name for leg in self.legs]
        if len(set(names)) != len(names):
            raise ValueError("duplicate leg names")
        if any(leg.weight <= 0 for leg in self.legs):
            raise ValueError("leg weights must be positive")
        if abs(sum(leg.weight for leg in self.legs) - 1.0) > 1e-6:
            raise ValueError("leg weights must sum to 1")
        if any("BTC" in leg.execution_symbol.upper() for leg in self.legs):
            raise ValueError("BTC contracts are reserved for the scalp bot; basket must not trade them")
        if not (0 < self.basket_multiplier <= 3):
            raise ValueError("basket_multiplier must be in (0, 3]")
        if not (1 <= int(self.leverage) <= 20):
            raise ValueError("leverage must be in [1, 20]")
        if not (0 < self.max_basket_margin_frac <= 0.5):
            raise ValueError("max_basket_margin_frac must be in (0, 0.5]")
        if self.margin_mode not in {"isolated", "cross"}:
            raise ValueError("margin_mode must be isolated or cross")


# --------------------------------------------------------------------------- 纯函数（便于测试）


def completed_daily_bars(df: pd.DataFrame, now_utc: datetime, close_buffer_minutes: int = 15) -> pd.DataFrame:
    """丢掉尚未收盘的当日 bar（Yahoo 盘中会返回正在形成的日线）。

    以纽约时间判断：bar 的交易日 < 今天 → 已收盘；== 今天 → 只有过了 16:00+buffer 才算收盘。
    """
    if df.empty:
        return df
    now_ny = now_utc.astimezone(NEW_YORK)
    trade_day = pd.to_datetime(df["date"], utc=True).dt.tz_convert(NEW_YORK).dt.date
    close_cutoff = now_ny.replace(hour=16, minute=0, second=0, microsecond=0) + timedelta(minutes=close_buffer_minutes)
    today_done = now_ny >= close_cutoff
    keep = (trade_day < now_ny.date()) | ((trade_day == now_ny.date()) & today_done)
    return df[keep.values].reset_index(drop=True)


def leg_signal(df: pd.DataFrame) -> dict[str, Any]:
    """最新已收盘 bar 的投票结果。"""
    close = pd.Series(df["close"].astype(float).values, index=pd.to_datetime(df["date"], utc=True))
    target = vote_target(close)
    parts = vote_components(close)
    last_ts = close.index[-1]
    return {
        "signal_date": str(last_ts.tz_convert(NEW_YORK).date()),
        "close": float(close.iloc[-1]),
        "vote": round(float(target.iloc[-1]), 4),
        "rules": {name: int(parts[name].iloc[-1]) for name in VOTE_RULES},
        "bars": int(len(close)),
    }


def plan_leg_orders(
    *,
    equity: float,
    legs: list[BasketLeg],
    votes: dict[str, float],
    positions: dict[str, dict[str, float]],
    prices: dict[str, float],
    contract_sizes: dict[str, float],
    min_contracts: dict[str, float],
    contract_steps: dict[str, float],
    multiplier: float,
    leverage: float,
    max_margin_frac: float,
    band_frac: float,
    min_order_notional: float,
) -> dict[str, Any]:
    """计算每条腿的目标名义与下单合约数（正 = 加多，负 = 减多）。"""
    raw = {leg.name: multiplier * equity * leg.weight * max(0.0, min(1.0, votes[leg.name])) for leg in legs}
    total_target = sum(raw.values())
    margin_cap_notional = max_margin_frac * equity * leverage
    scale = 1.0
    if total_target > margin_cap_notional > 0:
        scale = margin_cap_notional / total_target
    plans: dict[str, Any] = {}
    for leg in legs:
        name = leg.name
        target = raw[name] * scale
        price = prices[name]
        csize = contract_sizes[name]
        cur_contracts = float(positions.get(name, {}).get("contracts", 0.0))
        cur_notional = cur_contracts * csize * price
        full_leg = multiplier * equity * leg.weight
        diff = target - cur_notional
        step = contract_steps[name]
        if target <= 0:
            delta = -cur_contracts  # 投票清零：全平
        else:
            delta_raw = diff / (csize * price)
            delta = math.copysign(math.floor(abs(delta_raw) / step) * step, delta_raw)
        reason = "rebalance"
        if delta == 0:
            reason = "no_change"
        elif target > 0 and abs(diff) < band_frac * full_leg:
            delta, reason = 0.0, "within_band"
        elif abs(delta) * csize * price < min_order_notional and target > 0:
            delta, reason = 0.0, "below_min_notional"
        elif delta > 0 and delta < min_contracts[name]:
            delta, reason = 0.0, "below_min_size"
        plans[name] = {
            "symbol": leg.execution_symbol,
            "vote": votes[name],
            "target_notional": round(target, 2),
            "current_notional": round(cur_notional, 2),
            "current_contracts": cur_contracts,
            "delta_contracts": round(delta, 8),
            "price": price,
            "reason": reason,
        }
    return {"equity": round(equity, 2), "margin_scale": round(scale, 4), "legs": plans}


# --------------------------------------------------------------------------- 执行器


class TrendBasketExecutor:
    def __init__(
        self,
        config: TrendBasketConfig,
        *,
        client: OkxClient | None = None,
        fetch_history: Callable[[BasketLeg, int], pd.DataFrame] | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ):
        self.config = config
        credentials = None
        if config.api_key and config.api_secret and config.api_passphrase:
            credentials = OkxCredentials(config.api_key, config.api_secret, config.api_passphrase)
        self.client = client or OkxClient(credentials, trading_mode=config.mode, proxy=config.proxy)
        self.store = StateStore(config.state_db_path)
        self._fetch_history = fetch_history or self._fetch_yahoo_history
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._markets: dict[str, Any] | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "TrendBasketExecutor":
        p = Path(path)
        return cls(TrendBasketConfig.from_dict(json.loads(p.read_text()), base_dir=Path.cwd()))

    # ---- 日志 / 状态
    def _log(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = {"event": event, "time": self._now().isoformat(timespec="seconds"), **payload}
        print(json.dumps(record, ensure_ascii=False, default=str), flush=True)
        self.store.append_action(record["time"], event.upper(), record)
        return record

    def _load_state(self) -> dict[str, Any]:
        raw = self.store.get_value(STATE_KEY)
        return json.loads(raw) if raw else {}

    def _save_state(self, state: dict[str, Any]) -> None:
        self.store.set_value(STATE_KEY, json.dumps(state, ensure_ascii=False, default=str))

    # ---- 数据
    def _fetch_yahoo_history(self, leg: BasketLeg, days: int) -> pd.DataFrame:
        import requests

        from scripts.fetch_public_etf_history import fetch_chunk, merge_history, normalize_dataframe

        cache = Path(self.config.signal_data_dir) / f"{leg.signal_symbol.upper()}-1d.feather"
        cache.parent.mkdir(parents=True, exist_ok=True)
        end = self._now()
        start = end - timedelta(days=days)
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        fetched = fetch_chunk(
            session,
            leg.signal_symbol,
            int(start.timestamp() * 1000),
            int(end.timestamp() * 1000),
            "1d",
            proxy=None,
            timeout_seconds=30.0,
        )
        existing = normalize_dataframe(pd.read_feather(cache)) if cache.exists() else fetched.iloc[0:0]
        merged = merge_history(existing, fetched, "1d")
        merged.reset_index(drop=True).to_feather(cache)
        return merged

    def _markets_info(self) -> dict[str, Any]:
        if self._markets is None:
            self._markets = self.client.load_markets()
        return self._markets

    def _account_equity(self) -> tuple[float, float]:
        balance = self.client.fetch_balance()
        info = (balance.get("info") or {}).get("data") or []
        total_eq = float(info[0].get("totalEq") or 0.0) if info else 0.0
        usdt = balance.get("USDT") or {}
        if total_eq <= 0:
            total_eq = float(usdt.get("total") or 0.0)
        free = float(usdt.get("free") or 0.0)
        return total_eq, free

    def _positions(self) -> dict[str, dict[str, float]]:
        symbols = [leg.execution_symbol for leg in self.config.legs]
        by_symbol = {leg.execution_symbol: leg.name for leg in self.config.legs}
        out: dict[str, dict[str, float]] = {}
        for pos in self.client.fetch_positions(symbols):
            name = by_symbol.get(pos.get("symbol"))
            contracts = float(pos.get("contracts") or 0.0)
            if name is None or contracts == 0:
                continue
            if pos.get("side") != "long":
                raise RuntimeError(f"unexpected non-long basket position on {pos.get('symbol')}: {pos.get('side')}")
            entry = out.setdefault(name, {"contracts": 0.0, "margin": 0.0})
            entry["contracts"] += contracts
            entry["margin"] += float(pos.get("initialMargin") or 0.0)
        return out

    # ---- 主流程
    def bootstrap(self) -> dict[str, Any]:
        markets = self._markets_info()
        legs = {}
        for leg in self.config.legs:
            m = markets.get(leg.execution_symbol)
            if m is None:
                raise RuntimeError(f"market not found: {leg.execution_symbol}")
            legs[leg.name] = {
                "symbol": leg.execution_symbol,
                "contract_size": m.get("contractSize"),
                "min_amount": (m.get("limits") or {}).get("amount", {}).get("min"),
                "amount_step": (m.get("precision") or {}).get("amount"),
            }
            if not self.config.dry_run:
                self.client.set_leverage(int(self.config.leverage), leg.execution_symbol, self.config.margin_mode, "long")
        return self._log(
            "bootstrap",
            {"mode": self.config.mode, "dry_run": self.config.dry_run, "enabled": self.config.enabled, "legs": legs},
        )

    def evaluate_once(self, *, force: bool = False) -> dict[str, Any]:
        cfg = self.config
        state = self._load_state()
        now = self._now()
        signals: dict[str, dict[str, Any]] = {}
        for leg in cfg.legs:
            df = completed_daily_bars(self._fetch_history(leg, cfg.history_days), now, cfg.ny_close_buffer_minutes)
            if len(df) < 200:
                return self._log("error", {"reason": "insufficient_history", "leg": leg.name, "bars": len(df)})
            signals[leg.name] = leg_signal(df)
        signal_key = "|".join(f"{n}:{s['signal_date']}" for n, s in sorted(signals.items()))

        if not cfg.enabled:
            return self._log("disabled", {"signals": signals})
        retry_after = float(state.get("retry_after_ts") or 0.0)
        same_signal = state.get("last_signal_key") == signal_key and bool(state.get("last_dry_run")) == cfg.dry_run
        if not force and same_signal and state.get("last_status") == "ok":
            return {"event": "hold", "signal_key": signal_key}
        if not force and state.get("last_status") == "error" and now.timestamp() < retry_after:
            return {"event": "retry_wait", "retry_after": retry_after}

        markets = self._markets_info()
        equity, free = self._account_equity()
        positions = self._positions()
        prices, csizes, mins, steps = {}, {}, {}, {}
        for leg in cfg.legs:
            m = markets[leg.execution_symbol]
            ticker = self.client.exchange.fetch_ticker(leg.execution_symbol)
            prices[leg.name] = float(ticker.get("last") or ticker.get("close"))
            csizes[leg.name] = float(m.get("contractSize") or 1.0)
            mins[leg.name] = float((m.get("limits") or {}).get("amount", {}).get("min") or 0.0)
            steps[leg.name] = float((m.get("precision") or {}).get("amount") or 1.0)
        plan = plan_leg_orders(
            equity=equity,
            legs=cfg.legs,
            votes={n: s["vote"] for n, s in signals.items()},
            positions=positions,
            prices=prices,
            contract_sizes=csizes,
            min_contracts=mins,
            contract_steps=steps,
            multiplier=cfg.basket_multiplier,
            leverage=cfg.leverage,
            max_margin_frac=cfg.max_basket_margin_frac,
            band_frac=cfg.rebalance_band_frac,
            min_order_notional=cfg.min_order_notional_usdt,
        )
        plan["free_usdt"] = round(free, 2)
        results: dict[str, Any] = {}
        status = "ok"
        # 先减仓后加仓，释放的保证金可给加仓用。
        ordered = sorted(cfg.legs, key=lambda leg: plan["legs"][leg.name]["delta_contracts"])
        for leg in ordered:
            lp = plan["legs"][leg.name]
            delta = float(lp["delta_contracts"])
            if delta == 0:
                continue
            if cfg.dry_run:
                results[leg.name] = {"status": "dry_run", "delta_contracts": delta}
                continue
            try:
                results[leg.name] = self._place(leg, delta)
            except Exception as exc:  # 单腿失败不影响另一腿；整体标记 error 并按 failure_retry_seconds 重试
                status = "error"
                results[leg.name] = {"status": "error", "error": str(exc)}
        state.update(
            {
                "last_signal_key": signal_key,
                "last_dry_run": cfg.dry_run,
                "last_status": status,
                "last_run": now.isoformat(timespec="seconds"),
                "retry_after_ts": now.timestamp() + cfg.failure_retry_seconds if status == "error" else 0.0,
                "signals": signals,
                "plan": plan,
                "results": results,
            }
        )
        self._save_state(state)
        return self._log("rebalance", {"status": status, "dry_run": cfg.dry_run, "signals": signals, "plan": plan, "results": results})

    def _place(self, leg: BasketLeg, delta: float) -> dict[str, Any]:
        cfg = self.config
        params: dict[str, Any] = {"tdMode": cfg.margin_mode, "posSide": "long"}
        if delta > 0:
            side = "buy"
        else:
            side = "sell"
            params["reduceOnly"] = True
        amount = float(self.client.exchange.amount_to_precision(leg.execution_symbol, abs(delta)))
        order = self.client.create_order(leg.execution_symbol, "market", side, amount, params=params)
        return {"status": "ok", "side": side, "amount": amount, "order_id": order.get("id")}

    def run_loop(self) -> None:
        self.bootstrap()
        while True:
            try:
                self.evaluate_once()
            except Exception as exc:
                self._log("error", {"reason": "evaluate_exception", "error": str(exc)})
            time.sleep(max(30, int(self.config.poll_interval_seconds)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Trend-vote basket leg (XAU / QQQ perps)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-loop", action="store_true")
    parser.add_argument("--evaluate-once", action="store_true")
    parser.add_argument("--force", action="store_true", help="忽略去重，立即按当前信号再平衡一次")
    args = parser.parse_args()
    executor = TrendBasketExecutor.from_file(args.config)
    if args.run_loop:
        executor.run_loop()
        return
    executor.bootstrap()
    if args.evaluate_once:
        print(json.dumps(executor.evaluate_once(force=args.force), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
