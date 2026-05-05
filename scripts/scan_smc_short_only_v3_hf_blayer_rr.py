#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reproduce_smc_short_only_v1_10x import main as reproduce_main  # noqa: E402


DEFAULT_OUTPUT = ROOT / "var" / "pa_ict_liquidity" / "smc_short_only_v3_hf_blayer_rr_scan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan fixed target RR for short-only v3 high-frequency B-layer.")
    parser.add_argument("--rr-values", default="1.25,1.5,1.75,2.0,2.25")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def run_one(rr: float) -> dict[str, Any]:
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "reproduce_smc_short_only_v3_hf_blayer_10x.py"),
        "--target-rr",
        str(rr),
        "--output",
        tmp_path,
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    data = json.loads(Path(tmp_path).read_text())
    Path(tmp_path).unlink(missing_ok=True)
    y2026 = data.get("yearly", {}).get("2026", {})
    w60 = data.get("windows", {}).get("last_60d", {})
    w30 = data.get("windows", {}).get("last_30d", {})
    result = {
        "target_rr": rr,
        "overall": data["overall"],
        "2026": y2026,
        "60d": w60,
        "30d": w30,
        "execution_summary": data["execution_summary"],
        "risk_diagnostics": data["risk_diagnostics"],
    }
    result["score"] = round(
        float(result["overall"]["total_return_pct"])
        - float(result["overall"]["max_drawdown_pct"]) * 5.0
        + float(y2026.get("total_return_pct", 0.0)) * 2.0
        + float(w60.get("total_return_pct", 0.0)),
        4,
    )
    return result


def main() -> None:
    args = parse_args()
    rr_values = [float(item.strip()) for item in args.rr_values.split(",") if item.strip()]
    rows = [run_one(rr) for rr in rr_values]
    rows.sort(key=lambda item: item["score"], reverse=True)
    report = {"top": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)
    for idx, item in enumerate(rows, start=1):
        overall = item["overall"]
        y2026 = item["2026"]
        w60 = item["60d"]
        w30 = item["30d"]
        print(
            f"{idx:02d} rr={item['target_rr']:.2f} "
            f"full={overall['total_return_pct']:.2f}%/{overall['max_drawdown_pct']:.2f}% "
            f"2026={y2026.get('total_return_pct', 0.0):.2f}% "
            f"60d={w60.get('total_return_pct', 0.0):.2f}% "
            f"30d={w30.get('total_return_pct', 0.0):.2f}% "
            f"trades={overall['trades']} score={item['score']:.2f}"
        )


if __name__ == "__main__":
    main()
