#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SECRET_KEYS = (
    "api_key",
    "api_secret",
    "api_passphrase",
    "telegram_token",
    "telegram_chat_id",
    "proxy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create high-leverage live config from template and existing live secrets.")
    parser.add_argument("--source", default="config/config.live.5x-3pct.json")
    parser.add_argument("--template", default="config/config.live.high-leverage-structure.template.json")
    parser.add_argument("--target", default="config/config.live.high-leverage-structure.json")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    template_path = Path(args.template)
    target_path = Path(args.target)

    if not source_path.exists():
        raise FileNotFoundError(f"missing source live config: {source_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"missing high leverage template: {template_path}")

    source = load_json(source_path)
    target = load_json(template_path)
    for key in SECRET_KEYS:
        if source.get(key) is not None:
            target[key] = source[key]
    if "telegram_enabled" in source:
        target["telegram_enabled"] = bool(source["telegram_enabled"])

    missing = [key for key in ("api_key", "api_secret", "api_passphrase") if not target.get(key)]
    if missing:
        raise ValueError(f"missing live credentials after merge: {', '.join(missing)}")

    target_path.write_text(json.dumps(target, ensure_ascii=False, indent=2) + "\n")
    print(f"prepared_high_leverage_live_config target={target_path}")


if __name__ == "__main__":
    main()
