#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

PAPER_SRC = ROOT / "config" / "config.paper.high-leverage-structure.json"
LIVE_TEMPLATE_SRC = ROOT / "config" / "config.live.high-leverage-structure.template.json"

PAPER_A = ROOT / "config" / "config.paper.high-leverage-structure.fallback-a.json"
PAPER_B = ROOT / "config" / "config.paper.high-leverage-structure.fallback-b.json"
PAPER_C = ROOT / "config" / "config.paper.high-leverage-structure.fallback-c.json"

LIVE_A = ROOT / "config" / "config.live.high-leverage-structure.fallback-a.template.json"
LIVE_B = ROOT / "config" / "config.live.high-leverage-structure.fallback-b.template.json"
LIVE_C = ROOT / "config" / "config.live.high-leverage-structure.fallback-c.template.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fallback_a(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["enable_smc_short_live"] = False
    return out


def _defensive_bucket_rules() -> list[dict[str, Any]]:
    return [
        {
            "name": "n3_b9_b6_conflict_target2",
            "net_eq": 3,
            "bull_eq": 9,
            "bear_eq": 6,
            "conflict_mode": "conflict",
            "target_effective_leverage": 2.0,
        },
        {
            "name": "bear_total_6_conflict_defense_cap2",
            "bear_eq": 6,
            "conflict_mode": "conflict",
            "target_effective_leverage": 2.0,
        },
        {
            "name": "nbb_6_11_5_conflict_target2",
            "net_eq": 6,
            "bull_eq": 11,
            "bear_eq": 5,
            "conflict_mode": "conflict",
            "target_effective_leverage": 2.0,
        },
    ]


def fallback_b(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["enable_smc_short_live"] = False
    out["enable_long_score_bucket_sizing_live"] = True
    out["long_score_bucket_sizing_rules"] = _defensive_bucket_rules()
    return out


def fallback_c(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["enable_smc_short_live"] = False
    out["enable_gap_smc_short_live"] = False
    out["enable_long_score_bucket_sizing_live"] = False
    return out


def main() -> None:
    sources = [
        (PAPER_SRC, PAPER_A, PAPER_B, PAPER_C),
        (LIVE_TEMPLATE_SRC, LIVE_A, LIVE_B, LIVE_C),
    ]
    for src, out_a, out_b, out_c in sources:
        base = load_json(src)
        dump_json(out_a, fallback_a(base))
        dump_json(out_b, fallback_b(base))
        dump_json(out_c, fallback_c(base))
        print(src)
        print(" ->", out_a)
        print(" ->", out_b)
        print(" ->", out_c)


if __name__ == "__main__":
    main()
