#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.nasdaq100_cn_strategy_utils import ROOT, latest_decision, load_config, load_strategy_frame, run_full_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the latest paper plan for the CN Nasdaq-100 ETF driven by QQQ signals.")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.paper.cn-nasdaq100-etf.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    frame = load_strategy_frame(config)
    decision = latest_decision(frame, config)
    summary = run_full_strategy(frame, config)
    payload = {
        "config": str(Path(args.config).resolve()),
        "decision": decision,
        "summary": {
            "total_return_pct": summary["total_return_pct"],
            "max_drawdown_pct": summary["max_drawdown_pct"],
            "yearly_returns_pct": summary["yearly_returns_pct"],
            "trades": summary["trades"],
        },
    }
    output_path = ROOT / str(config["paper_decisions_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
