from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.strategy_router import StrategyRouter
from bot.router_executor import StrategyRouterExecutionEngine


def _default_config_path() -> str:
    live = ROOT / "config" / "config.live.strategy-router.json"
    if live.exists():
        return str(live.relative_to(ROOT))
    return "config/config.live.strategy-router.template.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BTC vs QQQ/USDT route and persist selected strategy state.")
    parser.add_argument("--config", default=_default_config_path())
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--evaluate-once", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-loop", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    return parser.parse_args()


def _print_output(payload: dict[str, object], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> dict | None:
    args = parse_args()
    if args.execute:
        engine = StrategyRouterExecutionEngine.from_file(Path(args.config))
        if args.run_loop:
            engine.run_loop(poll_interval_seconds=args.poll_interval_seconds)
            return None
        if args.evaluate_once:
            status = engine.evaluate_latest()
        else:
            status = engine.bootstrap()
        _print_output(status, args.json)
        return status

    router = StrategyRouter.from_file(Path(args.config))
    if args.run_loop:
        router.run_loop(poll_interval_seconds=args.poll_interval_seconds)
        return None
    status = router.evaluate_latest()
    _print_output(status, args.json)
    return status


if __name__ == "__main__":
    main()
