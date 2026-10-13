from collections import Counter

import pytest

from pcb_inspection.dataset.splitting import assign_captures, find_leakage, split_boards

RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


def boards(n, stratum="all"):
    return {f"GOOD_{i:03d}": stratum for i in range(1, n + 1)}


def test_each_board_in_exactly_one_split_with_expected_counts():
    split = split_boards(boards(10), RATIOS, seed=0)
    assert set(split) == set(boards(10))
    assert Counter(split.values()) == {"train": 6, "val": 2, "test": 2}


def test_remainder_allocation_keeps_total():
    split = split_boards(boards(7), RATIOS, seed=0)
    assert len(split) == 7
    assert sum(Counter(split.values()).values()) == 7


def test_deterministic_and_independent_of_input_order():
    forward = boards(20)
    backward = dict(reversed(list(forward.items())))
    assert split_boards(forward, RATIOS, seed=3) == split_boards(backward, RATIOS, seed=3)


def test_stratified_split_covers_every_stratum():
    strata = {**boards(10, "A|good"), **{f"GOOD_{i:03d}": "A|defective" for i in range(11, 21)}}
    split = split_boards(strata, RATIOS, seed=1)
    for stratum in ("A|good", "A|defective"):
        counts = Counter(split[b] for b, s in strata.items() if s == stratum)
        assert counts == {"train": 6, "val": 2, "test": 2}


@pytest.mark.parametrize("ratios", [{}, {"train": 0.5, "test": 0.4}, {"train": 1.2, "test": -0.2}])
def test_invalid_ratios(ratios):
    with pytest.raises(ValueError):
        split_boards(boards(5), ratios, seed=0)


def test_captures_follow_their_board_and_no_leakage():
    board_split = split_boards(boards(5), RATIOS, seed=0)
    capture_boards = {f"{b}_TOP_S20261002-01_C00{k}.png": b for b in board_split for k in (1, 2, 3)}
    capture_split = assign_captures(capture_boards, board_split)
    assert all(capture_split[c] == board_split[b] for c, b in capture_boards.items())
    assert find_leakage(capture_split, capture_boards) == {}


def test_leakage_is_detected():
    capture_boards = {"a.png": "GOOD_001", "b.png": "GOOD_001", "c.png": "DEFECT_001"}
    capture_split = {"a.png": "train", "b.png": "test", "c.png": "train"}
    assert find_leakage(capture_split, capture_boards) == {"GOOD_001": {"train", "test"}}


def test_assign_captures_requires_every_board():
    with pytest.raises(KeyError):
        assign_captures({"a.png": "GOOD_009"}, {"GOOD_001": "train"})
