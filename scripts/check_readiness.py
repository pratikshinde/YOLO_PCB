"""Report project readiness: gates and their blockers, open clarifications, unknown values.

1. Readiness gates (configs/project.yaml, readiness_gates): READY or BLOCKED,
   with every blocking value and its question ID where one exists.
2. Answers marked NEEDS CLARIFICATION (recorded verbatim).
3. Number of unknown (null) values per config section; --list-unknowns prints them all.
4. Images present in the capture folders.

Usage:  python scripts/check_readiness.py [--profile <model>/<variant>] [--list-unknowns]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_inspection.config import find_unresolved, gate_blockers, load_config, load_profile  # noqa: E402
from pcb_inspection.dataset.naming import CONDITIONS  # noqa: E402


def print_unknowns(label: str, config: dict[str, Any], list_all: bool) -> None:
    unresolved = find_unresolved(config)
    counts = Counter(path.split(".")[0] for path in unresolved)
    summary = ", ".join(f"{section} {n}" for section, n in counts.items())
    print(f"  {label}: {len(unresolved)} ({summary})")
    if list_all:
        for path in unresolved:
            print(f"      - {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report project readiness.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "project.yaml")
    parser.add_argument("--profile", help="profile id, default: profiles.active in the config")
    parser.add_argument("--list-unknowns", action="store_true", help="print every unknown (null) path")
    args = parser.parse_args()

    config = load_config(args.config)
    profile_id = args.profile or config["profiles"]["active"]
    profile = load_profile(ROOT / config["profiles"]["directory"], profile_id)
    clarifications = config.get("needs_clarification", []) + profile.get("needs_clarification", [])
    clarified_by = {
        (f"profile:{path}" if item in profile.get("needs_clarification", []) else path): item["id"]
        for item in clarifications
        for path in item["affects"]
    }

    print(f"Profile: {profile_id}\n")
    print("== Readiness gates ==")
    for name, gate in config["readiness_gates"].items():
        blockers = gate_blockers(gate["requires"], config, profile)
        print(f"  {'READY' if not blockers else 'BLOCKED':<8} {name}: {gate['description']}")
        for ref in blockers:
            note = f"   ({clarified_by[ref]} NEEDS CLARIFICATION)" if ref in clarified_by else ""
            print(f"             - {ref}{note}")

    print(f"\n== NEEDS CLARIFICATION: {len(clarifications)} ==")
    for item in clarifications:
        holds_up = f"; holds up: {', '.join(item['blocks'])}" if item.get("blocks") else ""
        print(f"  {item['id']} ({item.get('status', 'needs_clarification')}{holds_up})")
        reported = item["reported"] if isinstance(item["reported"], list) else [item["reported"]]
        for answer in reported:
            print(f"      reported: {answer!r}")
        for point in item["clarify"]:
            print(f"      clarify:  {point}")

    print("\n== Unknown (null) values ==")
    print_unknowns(args.config.name, config, args.list_unknowns)
    print_unknowns(f"profile {profile_id}", profile, args.list_unknowns)
    if not args.list_unknowns:
        print("  (run with --list-unknowns to print every path)")

    data_root = ROOT / config["data"]["root"]
    print("\n== Captured images ==")
    folders = [("experiment_0 (phone)", data_root / "experiment_0" / "images")]
    folders += [(f"raw/{c}", data_root / "raw" / c) for c in CONDITIONS]
    for label, base in folders:
        files = [p for p in base.rglob("*") if p.is_file() and not p.name.startswith(".")] if base.is_dir() else []
        print(f"  {label:<22} {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
