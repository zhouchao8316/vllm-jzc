# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks for measured graph-tier targets and safe phase movement."""

import pytest

from vllm.v1.core.sched.decode_tiers import (
    minimum_decode_phase_delays,
    select_decode_tier_split,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("active", "expected"),
    [(8, (4, 4)), (9, (5, 4)), (10, (6, 4)), (11, (7, 4)),
     (12, (8, 4)), (13, (8, 5))],
)
def test_reproduces_measured_targets(active, expected):
    assert select_decode_tier_split(active, (4, 8, 12)) == expected


@pytest.mark.parametrize("active", range(2, 9))
def test_small_cohorts_keep_existing_balanced_policy(active):
    assert select_decode_tier_split(active, (4, 8)) == (
        (active + 1) // 2, active // 2
    )


def test_uses_actual_capture_sizes_instead_of_assuming_all_multiples_of_four():
    assert select_decode_tier_split(12, (4, 8, 12)) == (8, 4)
    assert select_decode_tier_split(12, (4, 6, 8, 12)) == (6, 6)
    assert select_decode_tier_split(13, (4, 8)) == (8, 5)


@pytest.mark.parametrize("sizes", [(), (0,), (-4, 0), (4,)])
def test_does_not_invent_uncaptured_graphs(sizes):
    assert select_decode_tier_split(13, sizes) is None


@pytest.mark.parametrize("active", [0, 1])
def test_does_not_split_a_single_request(active):
    assert select_decode_tier_split(active, (4, 8, 12)) is None


@pytest.mark.parametrize("active", range(2, 25))
def test_targets_preserve_all_requests_and_fit_captured_graphs(active):
    heavy, light = select_decode_tier_split(active, (4, 8, 12))
    assert heavy >= light >= 1
    assert heavy + light == active
    assert heavy <= 12


@pytest.mark.parametrize(
    ("due", "waiting", "target", "expected"),
    [(1, 11, (8, 4), (0, 3)),
     (1, 12, (8, 5), (0, 4)),
     (8, 0, (4, 4), (4, 0)),
     (0, 12, (8, 4), (0, 4)),
     (4, 8, (8, 4), None),
     (8, 4, (8, 4), None),
     (7, 5, (8, 4), (0, 1))],
)
def test_rebalances_by_minimum_delays_without_pulling_requests_forward(
    due, waiting, target, expected
):
    assert minimum_decode_phase_delays(due, waiting, *target) == expected


def test_delay_count_is_minimal_for_every_phase_layout():
    for active in range(2, 25):
        target = select_decode_tier_split(active, (4, 8, 12))
        for due in range(active + 1):
            waiting = active - due
            delays = minimum_decode_phase_delays(due, waiting, *target)
            if sorted((due, waiting)) == sorted(target):
                assert delays is None
                continue
            delay_due, delay_next = delays
            assert 0 <= delay_due <= due
            assert 0 <= delay_next <= waiting
            after = due - delay_due + delay_next, waiting + delay_due - delay_next
            assert sorted(after) == sorted(target)
            possible = [
                x + y
                for x in range(due + 1)
                for y in range(waiting + 1)
                if sorted((due - x + y, waiting + x - y)) == sorted(target)
            ]
            assert delay_due + delay_next == min(possible)


@pytest.mark.parametrize(
    "args", [(-1, 5, 4, 0), (3, 4, 4, 4), (4, 4, 8, 0)]
)
def test_invalid_phase_layout_is_left_unchanged(args):
    assert minimum_decode_phase_delays(*args) is None
