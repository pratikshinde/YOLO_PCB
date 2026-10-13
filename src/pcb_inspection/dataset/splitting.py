"""Dataset splitting by physical board identity.

All captures of one physical board (every side, session and condition) go to the
same split, so a model is never evaluated on a board it saw during training or
tuning. Splitting individual images at random would leak board appearance
between splits and overstate performance.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping


def split_boards(board_strata: Mapping[str, str], ratios: Mapping[str, float], seed: int) -> dict[str, str]:
    """Assign every board ID to exactly one split.

    board_strata maps board_id -> stratum key (e.g. "VARIANT_A|defective"). Boards
    of each stratum are distributed over the splits according to ratios, so every
    split gets a share of each variant/condition. Pass the same key for all boards
    to disable stratification. Small strata may leave a split without boards of
    that stratum; check the result.
    """
    if not ratios:
        raise ValueError("ratios must not be empty")
    if any(r <= 0 for r in ratios.values()):
        raise ValueError(f"ratios must be positive, got {dict(ratios)}")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"ratios must sum to 1, got {sum(ratios.values())}")

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for board_id, stratum in board_strata.items():
        by_stratum[stratum].append(board_id)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        boards = sorted(by_stratum[stratum])
        rng.shuffle(boards)
        start = 0
        for split, count in zip(ratios, _allocate(len(boards), list(ratios.values()))):
            for board_id in boards[start:start + count]:
                assignment[board_id] = split
            start += count
    return assignment


def _allocate(n: int, ratios: list[float]) -> list[int]:
    """Largest-remainder allocation of n items over ratios; ties go to the earlier split."""
    exact = [n * r for r in ratios]
    counts = [math.floor(x) for x in exact]
    by_remainder = sorted(range(len(ratios)), key=lambda i: (counts[i] - exact[i], i))
    for i in by_remainder[: n - sum(counts)]:
        counts[i] += 1
    return counts


def assign_captures(capture_boards: Mapping[str, str], board_split: Mapping[str, str]) -> dict[str, str]:
    """Map each capture to the split of its board (capture_boards: capture -> board_id)."""
    unassigned = sorted({b for b in capture_boards.values() if b not in board_split})
    if unassigned:
        raise KeyError(f"boards without a split: {unassigned}")
    return {capture: board_split[board] for capture, board in capture_boards.items()}


def find_leakage(capture_split: Mapping[str, str], capture_boards: Mapping[str, str]) -> dict[str, set[str]]:
    """Return boards whose captures occur in more than one split. Empty means no leakage."""
    splits_of_board: dict[str, set[str]] = defaultdict(set)
    for capture, split in capture_split.items():
        splits_of_board[capture_boards[capture]].add(split)
    return {board: splits for board, splits in splits_of_board.items() if len(splits) > 1}
