from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from execution.okx_executor import OkxExecutionEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap OKX execution engine")
    parser.add_argument("--config", default="execution/config.example.json")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--evaluate-once", action="store_true")
    parser.add_argument("--run-loop", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=int, default=5)
    parser.add_argument("--close-buffer-seconds", type=int, default=5)
    return parser.parse_args()


def _print_output(payload: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> dict | None:
    args = parse_args()
    engine = OkxExecutionEngine.from_file(args.config)
    if args.run_loop:
        engine.run_loop(
            poll_interval_seconds=args.poll_interval_seconds,
            close_buffer_seconds=args.close_buffer_seconds,
        )
        return None
    if args.evaluate_once:
        status = engine.evaluate_latest()
    else:
        status = engine.bootstrap()
    _print_output(status, args.json)
    return status


if __name__ == "__main__":
    main()
