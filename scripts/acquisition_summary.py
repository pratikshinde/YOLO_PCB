"""Summary and consistency audit of the camera-acquisition data (data/acquisition, curated, quarantine).

    python scripts/acquisition_summary.py

Prints one line per session (type, board, stage, status, captures, review
counts) and per physical board, then audits the files: originals and review
copies must match their checksums, every copy must lie under its own session's
type (TEST isolation), and nothing in curated/ or quarantine/ may be
unaccounted for. Read-only. Exit code 1 if the audit finds issues.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcb_inspection.acquisition.session import AcquisitionStore, audit, settings_from_configs  # noqa: E402
from pcb_inspection.config import load_config, load_profile  # noqa: E402


def main() -> int:
    project = load_config(ROOT / "configs" / "project.yaml")
    capture = load_config(ROOT / "configs" / "capture" / "droidcam_e1b.yaml")
    profile = load_profile(ROOT / project["profiles"]["directory"], capture["profile"])
    store = AcquisitionStore(settings_from_configs(ROOT / project["data"]["root"], profile, capture))
    sessions = store.sessions()
    if not sessions:
        print("No acquisition sessions yet. Start one with: python scripts/acquire.py")
        return 0

    print(f"  {'session':<14} {'type':<7} {'board':<12} {'stage':<17} {'status':<12} "
          f"{'req':>4} {'ok':>4} {'fail':>4} {'kept':>5} {'rej':>4} {'pend':>5}")
    boards: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for s in sessions:
        c = s.counts
        print(f"  {s.id:<14} {s.dataset_type:<7} {s.board_id or '-':<12} {s.data['stage']:<17} {s.status:<12} "
              f"{s.data['requested_image_count']:>4} {c['successful_captures']:>4} {c['failed_captures']:>4} "
              f"{c['kept_count']:>5} {c['rejected_count']:>4} {c['pending_review_count']:>5}")
        entry = boards[(s.dataset_type, s.data["split_group"])]
        entry[0] += 1
        entry[1] += c["successful_captures"]
        entry[2] += c["kept_count"]
    print("\n  Physical boards (split groups; all images of one group must stay in one split):")
    for (dataset_type, group), (n_sessions, n_images, n_kept) in sorted(boards.items()):
        print(f"  {dataset_type:<7} {group:<28} sessions {n_sessions:>3}   images {n_images:>4}   kept {n_kept:>4}")

    issues = audit(store)
    print(f"\nAudit: {'OK' if not issues else f'{len(issues)} issue(s)'}")
    for issue in issues:
        print(f"  ISSUE {issue}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
