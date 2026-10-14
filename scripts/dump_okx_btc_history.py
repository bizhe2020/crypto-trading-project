#!/usr/bin/env python3
"""只读导出 OKX 账户的 BTC-USDT-SWAP 已平仓持仓与逐笔成交（审计用）。

只调用两个 GET 私有接口，绝不下单 / 改单 / 撤单 / 改杠杆：
  - GET /api/v5/account/positions-history  （交易所权威已平仓持仓，含手续费与资金费）
  - GET /api/v5/trade/fills-history         （逐笔成交，近 3 个月）

OKX 对这两个接口只保留约 3 个月数据，--since 早于保留期的部分不会返回。

输出 JSON 结构与 scripts/audit_btc_scalp_live_history.py 的 --okx 输入一致：
  {"meta": {...}, "positions_history": [...], "fills": [...]}

用法（在能访问 OKX 的服务器上、于临时目录运行，凭证从实盘配置只读加载，不落盘、不打印）:
    .venv/bin/python scripts/dump_okx_btc_history.py \
        --config config/config.live.high-leverage-structure.json \
        --since 2026-04-01 --out /root/scratch/okx_history.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

INST_ID = "BTC-USDT-SWAP"
PAGE_LIMIT = 100


def _to_ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def paginate(
    fetch_page: Callable[[dict[str, Any]], list[dict[str, Any]]],
    *,
    cursor_field: str,
    time_field: str,
    since_ms: int,
    max_pages: int = 200,
    pause_seconds: float = 0.6,
) -> list[dict[str, Any]]:
    """按 OKX `after` 游标向更早翻页，直到没有数据或越过 since。

    OKX 约定：`after=<cursor>` 返回比该游标更早的记录，结果按时间倒序。
    """
    rows: list[dict[str, Any]] = []
    params: dict[str, Any] = {}
    for _ in range(max_pages):
        page = fetch_page(dict(params))
        if not page:
            break
        rows.extend(page)
        oldest = page[-1]
        if int(oldest[time_field]) < since_ms or len(page) < PAGE_LIMIT:
            break
        params["after"] = oldest[cursor_field]
        time.sleep(pause_seconds)
    return [row for row in rows if int(row[time_field]) >= since_ms]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="含 api_key/api_secret/api_passphrase 的配置（只读）")
    parser.add_argument("--since", default="2026-04-01", help="UTC 日期 YYYY-MM-DD")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import ccxt  # 延迟导入：本地无 ccxt 时 --help 仍可用

    cfg = json.loads(Path(args.config).read_text())
    exchange = ccxt.okx({
        "apiKey": cfg["api_key"],
        "secret": cfg["api_secret"],
        "password": cfg["api_passphrase"],
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    since_ms = _to_ms(args.since)
    base = {"instType": "SWAP", "instId": INST_ID, "limit": str(PAGE_LIMIT)}

    def positions_page(extra: dict[str, Any]) -> list[dict[str, Any]]:
        return exchange.privateGetAccountPositionsHistory({**base, **extra}).get("data") or []

    def fills_page(extra: dict[str, Any]) -> list[dict[str, Any]]:
        return exchange.privateGetTradeFillsHistory({**base, **extra}).get("data") or []

    positions = paginate(positions_page, cursor_field="uTime", time_field="uTime", since_ms=since_ms)
    fills = paginate(fills_page, cursor_field="billId", time_field="ts", since_ms=since_ms)

    def _span(rows: list[dict[str, Any]], field: str) -> list[str | None]:
        if not rows:
            return [None, None]
        values = sorted(int(row[field]) for row in rows)
        return [datetime.fromtimestamp(v / 1000, tz=timezone.utc).isoformat() for v in (values[0], values[-1])]

    payload = {
        "meta": {
            "inst_id": INST_ID,
            "since_utc": args.since,
            "dumped_at_utc": datetime.now(timezone.utc).isoformat(),
            "endpoints": ["GET /api/v5/account/positions-history", "GET /api/v5/trade/fills-history"],
            "positions_count": len(positions),
            "positions_span_utime": _span(positions, "uTime"),
            "fills_count": len(fills),
            "fills_span_ts": _span(fills, "ts"),
            "note": "OKX 仅保留约 3 个月；早于保留期的记录不会出现在此文件中。",
        },
        "positions_history": positions,
        "fills": fills,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
