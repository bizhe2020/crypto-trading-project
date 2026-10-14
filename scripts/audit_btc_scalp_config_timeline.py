#!/usr/bin/env python3
"""BTC scalp 实盘配置快照审计：脱敏副本、与参考配置的逐键差异、备份时间线。

只读取本地快照目录（由运维从 Tokyo 拉取），不连服务器、不改任何配置。

用法:
    python3 scripts/audit_btc_scalp_config_timeline.py \
        --snapshot-dir var/tokyo_audit/btc_scalp_20261013 \
        --compare config/config.candidate.btc-scalp-optimized-20260921.json \
        --compare config/config.live.high-leverage-structure.template.json \
        --out-json var/tmp/btc_headroom/config_timeline.json

快照目录约定：
  - config.live.high-leverage-structure.json         当前实盘配置（含明文密钥，勿提交）
  - config_backups/<name>.bak_<label>_<YYYYMMDDTHHMMSSZ>  每次改动前的备份

备份语义：bak_<label>_<ts> 是在 <ts>（UTC）做 <label> 改动之前的那一版。
因此相邻两份备份的差异 = 这段时间内的全部改动（中间若有未备份的改动会被合并）。

敏感键（api_key / secret / passphrase / token / chat_id 等）的值永远不输出，
只标注 "<redacted>"，差异里也只报告“变了”而不报值。
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LIVE_CONFIG_NAME = "config.live.high-leverage-structure.json"
REDACTED = "<redacted>"
SENSITIVE_MARKERS = ("api_key", "apikey", "secret", "passphrase", "password", "token", "chat_id")
BACKUP_RE = re.compile(r"\.bak_(?:(?P<label>.+?)_)?(?P<ts>\d{8}T\d{6}Z)$")


def is_sensitive(key: str) -> bool:
    low = key.lower()
    return any(marker in low for marker in SENSITIVE_MARKERS)


def redact(value: Any) -> Any:
    """递归脱敏：敏感键的值替换为 <redacted>。"""
    if isinstance(value, dict):
        return {k: (REDACTED if is_sensitive(k) else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def diff_configs(left: dict, right: dict) -> list[dict]:
    """返回 left 与 right 逐键差异（顶层键）。敏感键只报告是否变化，不带值。"""
    rows: list[dict] = []
    for key in sorted(set(left) | set(right)):
        in_left, in_right = key in left, key in right
        lval, rval = left.get(key), right.get(key)
        if in_left and in_right and lval == rval:
            continue
        if is_sensitive(key):
            rows.append({"key": key, "left": REDACTED if in_left else "<missing>",
                         "right": REDACTED if in_right else "<missing>", "sensitive": True})
            continue
        rows.append({"key": key,
                     "left": lval if in_left else "<missing>",
                     "right": rval if in_right else "<missing>",
                     "sensitive": False})
    return rows


def parse_backup_name(name: str) -> tuple[str, datetime] | None:
    match = BACKUP_RE.search(name)
    if not match:
        return None
    ts = datetime.strptime(match.group("ts"), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return (match.group("label") or "(unlabeled)"), ts


def build_timeline(snapshot_dir: Path) -> list[dict]:
    """按备份时间排序，生成版本链及相邻版本差异。最后一版是当前实盘配置。"""
    backups: list[tuple[datetime, str, Path]] = []
    backup_dir = snapshot_dir / "config_backups"
    for path in sorted(backup_dir.glob("*.bak_*")) if backup_dir.exists() else []:
        parsed = parse_backup_name(path.name)
        if parsed is None:
            continue
        label, ts = parsed
        backups.append((ts, label, path))
    backups.sort(key=lambda item: item[0])

    versions: list[dict] = []
    for ts, label, path in backups:
        versions.append({
            "name": path.name,
            "superseded_at_utc": ts.isoformat(),
            "change_label_applied_after": label,
            "file_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            "config": json.loads(path.read_text()),
        })
    live_path = snapshot_dir / LIVE_CONFIG_NAME
    versions.append({
        "name": LIVE_CONFIG_NAME,
        "superseded_at_utc": None,
        "change_label_applied_after": None,
        "file_mtime_utc": datetime.fromtimestamp(live_path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "config": json.loads(live_path.read_text()),
    })

    timeline: list[dict] = []
    for prev, cur in zip(versions, versions[1:]):
        timeline.append({
            "from": prev["name"],
            "to": cur["name"],
            "change_at_utc": prev["superseded_at_utc"],
            "label": prev["change_label_applied_after"],
            "to_file_mtime_utc": cur["file_mtime_utc"],
            "changes": diff_configs(prev["config"], cur["config"]),
        })
    return timeline


def _fmt(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 120 else text[:117] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--compare", action="append", default=[],
                        help="参考配置路径，可重复；输出 live 相对每个参考的差异")
    parser.add_argument("--out-json", default="var/tmp/btc_headroom/config_timeline.json")
    parser.add_argument("--redacted-name", default="config.live.redacted.json",
                        help="写到快照目录下的脱敏副本文件名")
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    live = json.loads((snapshot_dir / LIVE_CONFIG_NAME).read_text())
    redacted_path = snapshot_dir / args.redacted_name
    redacted_path.write_text(json.dumps(redact(live), ensure_ascii=False, indent=2) + "\n")

    result: dict[str, Any] = {"snapshot_dir": str(snapshot_dir), "redacted_config": str(redacted_path),
                              "compare": {}, "timeline": build_timeline(snapshot_dir)}
    for ref in args.compare:
        ref_cfg = json.loads(Path(ref).read_text())
        result["compare"][ref] = diff_configs(ref_cfg, live)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")

    print(f"脱敏副本: {redacted_path}")
    for ref, rows in result["compare"].items():
        print(f"\n== 参考 {ref}  ->  实盘（{len(rows)} 个键不同）")
        for row in rows:
            print(f"  {row['key']}: {_fmt(row['left'])}  ->  {_fmt(row['right'])}")
    print("\n== 配置时间线（相邻版本差异）")
    for step in result["timeline"]:
        print(f"\n[{step['change_at_utc']}] {step['label']}  ({step['from']} -> {step['to']})")
        if not step["changes"]:
            print("  (无差异)")
        for row in step["changes"]:
            print(f"  {row['key']}: {_fmt(row['left'])}  ->  {_fmt(row['right'])}")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
