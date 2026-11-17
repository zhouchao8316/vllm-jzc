# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-tier targets and delay-only rebalancing for two decode phases."""

from bisect import bisect_left
from functools import lru_cache


@lru_cache(maxsize=128)
def select_decode_tier_split(
    num_requests: int, capture_sizes: tuple[int, ...]
) -> tuple[int, int] | None:
    """Choose a split by padded capacity, larger graph, then full graph slots.

    Small cohorts retain the balanced policy. This is a graph-capacity
    heuristic, not a device-time predictor.
    """
    sizes = sorted({size for size in capture_sizes if size > 0})
    if num_requests < 2 or not sizes or num_requests > 2 * sizes[-1]:
        return None
    if num_requests <= 2 * sizes[0]:
        return (num_requests + 1) // 2, num_requests // 2

    best: tuple[tuple[int, int, int, int], tuple[int, int]] | None = None
    for light in range(1, num_requests // 2 + 1):
        heavy = num_requests - light
        heavy_index = bisect_left(sizes, heavy)
        light_index = bisect_left(sizes, light)
        if heavy_index == len(sizes) or light_index == len(sizes):
            continue
        heavy_graph, light_graph = sizes[heavy_index], sizes[light_index]
        full_graphs = int(heavy == heavy_graph) + int(light == light_graph)
        score = (
            heavy_graph + light_graph,
            heavy_graph,
            -full_graphs,
            heavy - light,
        )
        candidate = score, (heavy, light)
        if best is None or candidate < best:
            best = candidate
    return None if best is None else best[1]


def minimum_decode_phase_delays(
    due: int, waiting: int, heavy: int, light: int
) -> tuple[int, int] | None:
    """Return the fewest one-step delays needed to reach either orientation.

    The first result delays due requests to the next step; the second delays
    next-step requests by one further step. Neither operation runs early.
    """
    if (
        min(due, waiting) < 0
        or min(heavy, light) < 1
        or due + waiting != heavy + light
        or sorted((due, waiting)) == sorted((heavy, light))
    ):
        return None
    candidates = []
    for first_phase in (heavy, light):
        difference = first_phase - due
        delay_due = max(0, -difference)
        delay_next = max(0, difference)
        if delay_due <= due and delay_next <= waiting:
            candidates.append((delay_due + delay_next, delay_due, delay_next))
    if not candidates:
        return None
    _, delay_due, delay_next = min(candidates)
    return delay_due, delay_next
