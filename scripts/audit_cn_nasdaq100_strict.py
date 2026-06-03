#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cn_nasdaq100_strict_utils import load_config, load_strict_frame, run_strict_path, summarize_path  # noqa: E402


def main() -> None:
    config_path = ROOT / "config" / "config.paper.cn-nasdaq100-etf.json"
    output_path = ROOT / "var" / "reports" / "cn_nasdaq100_strict_audit.json"
    config = load_config(config_path)
    config["conditional_leverage_enabled"] = False
    config["conditional_leverage_value"] = 1.0
    config["tiered_leverage_enabled"] = False
    config["tiered_leverage_rules"] = []
    frame = load_strict_frame(config)
    path = run_strict_path(frame, config)
    latest_rows = path.tail(5).copy()
    if not latest_rows.empty:
        latest_rows["date"] = latest_rows["date"].astype(str)
    payload = {
        "config_path": str(config_path),
        "summary": summarize_path(path),
        "latest_rows": latest_rows.to_dict(orient="records"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(output_path)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
