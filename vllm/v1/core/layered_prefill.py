# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared metadata and state helpers for layered prefill.

The scheduler must keep token progress and layer progress separate.  The
classes in this module intentionally contain no device or model code so that
they can safely cross the EngineCore/worker process boundary.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Mapping


DEFAULT_GROUP_TOKEN_TARGET = 512
DEFAULT_ALLOWED_NUM_GROUPS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class LayerGroupRange:
    """A contiguous, half-open range of global Transformer layers."""

    group_id: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.group_id < 0:
            raise ValueError("group_id must be non-negative")
        if self.start < 0 or self.end <= self.start:
            raise ValueError(
                f"invalid layer group range [{self.start}, {self.end})"
            )


@dataclass(frozen=True)
class LayeredPrefillConfig:
    """Configuration for the eager layered-prefill reference path.

    This is deliberately independent of Ascend configuration classes.  The
    same metadata is useful to upstream workers and the Ascend plugin, while
    ``enabled=False`` keeps the default path byte-for-byte compatible.
    """

    enabled: bool = False
    mode: str = "one_group"
    group_token_target: int = DEFAULT_GROUP_TOKEN_TARGET
    allowed_num_groups: tuple[int, ...] = DEFAULT_ALLOWED_NUM_GROUPS
    max_groups_per_step: int = 1
    require_pd_mixed: bool = True
    require_eager: bool = True

    def __post_init__(self) -> None:
        groups = tuple(sorted(set(int(v) for v in self.allowed_num_groups)))
        object.__setattr__(self, "allowed_num_groups", groups)
        if self.mode not in ("one_group", "adaptive"):
            raise ValueError(
                "layered_prefill_config.mode must be 'one_group' or 'adaptive'"
            )
        if self.group_token_target <= 0:
            raise ValueError("layered_prefill_config.group_token_target must be > 0")
        if not groups or groups[0] <= 0:
            raise ValueError(
                "layered_prefill_config.allowed_num_groups must contain positive values"
            )
        if self.max_groups_per_step <= 0:
            raise ValueError(
                "layered_prefill_config.max_groups_per_step must be > 0"
            )
        # Phase 1 has a single eager group per step.  Keep accepting adaptive
        # as a config spelling so a later phase can enable it without changing
        # the serialized plan, but reject k > 1 until its timing model exists.
        if self.enabled and self.max_groups_per_step != 1:
            raise ValueError(
                "Phase 1 layered prefill requires max_groups_per_step=1"
            )

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "LayeredPrefillConfig":
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        if not isinstance(additional_config, dict):
            return cls()
        scheduler_config = additional_config.get("scheduler_config", {})
        if not isinstance(scheduler_config, dict):
            return cls()
        raw = scheduler_config.get("layered_prefill_config", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(
                "additional_config.scheduler_config.layered_prefill_config "
                f"must be a dict, got {type(raw).__name__}"
            )
        allowed_num_groups = raw.get(
            "allowed_num_groups", DEFAULT_ALLOWED_NUM_GROUPS
        )
        if allowed_num_groups is None:
            allowed_num_groups = DEFAULT_ALLOWED_NUM_GROUPS
        return cls(
            enabled=bool(raw.get("enabled", False)),
            mode=str(raw.get("mode", "one_group")),
            group_token_target=int(
                raw.get("group_token_target", DEFAULT_GROUP_TOKEN_TARGET)
            ),
            allowed_num_groups=tuple(int(v) for v in allowed_num_groups),
            max_groups_per_step=int(raw.get("max_groups_per_step", 1)),
            require_pd_mixed=bool(raw.get("require_pd_mixed", True)),
            require_eager=bool(raw.get("require_eager", True)),
        )


def get_num_hidden_layers(vllm_config: Any) -> int:
    """Return the model's global layer count from supported config shapes."""

    model_config = getattr(vllm_config, "model_config", vllm_config)
    hf_config = getattr(model_config, "hf_text_config", None)
    if hf_config is None:
        hf_config = getattr(model_config, "hf_config", None)
    for config in (hf_config, model_config):
        value = getattr(config, "num_hidden_layers", None)
        if value is not None:
            return int(value)
    raise ValueError(
        "Layered prefill requires model_config.num_hidden_layers (or "
        "hf_text_config.num_hidden_layers)"
    )


def make_layer_group_ranges(
    num_hidden_layers: int, num_groups: int
) -> tuple[LayerGroupRange, ...]:
    """Partition layers into stable contiguous groups.

    The first groups receive one extra layer when the division is uneven.  This
    keeps every layer covered exactly once and makes a layout deterministic for
    graph keys and tests.
    """

    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be > 0")
    if num_groups <= 0 or num_groups > num_hidden_layers:
        raise ValueError(
            f"num_groups must be in [1, {num_hidden_layers}], got {num_groups}"
        )
    base, remainder = divmod(num_hidden_layers, num_groups)
    ranges: list[LayerGroupRange] = []
    start = 0
    for group_id in range(num_groups):
        end = start + base + (1 if group_id < remainder else 0)
        ranges.append(LayerGroupRange(group_id, start, end))
        start = end
    assert start == num_hidden_layers
    return tuple(ranges)


def make_pp_aligned_layer_group_ranges(
    num_hidden_layers: int, num_groups: int, pipeline_parallel_size: int
) -> tuple[LayerGroupRange, ...]:
    """Partition layers into groups that never cross a PP stage boundary.

    Layered Prefill keeps a prompt frontier on the stage that owns the active
    group.  Keeping group ranges inside the static PP partition makes that
    owner deterministic and avoids a reverse pipeline transfer.  Extra groups
    are distributed over stages in order, while every stage receives at least
    one group.
    """

    if pipeline_parallel_size <= 0:
        raise ValueError("pipeline_parallel_size must be > 0")
    if num_groups <= 0:
        raise ValueError("num_groups must be > 0")
    if pipeline_parallel_size == 1:
        return make_layer_group_ranges(num_hidden_layers, num_groups)
    if pipeline_parallel_size > num_hidden_layers:
        raise ValueError(
            "pipeline_parallel_size cannot exceed num_hidden_layers for "
            "layered prefill"
        )
    if num_groups < pipeline_parallel_size:
        num_groups = pipeline_parallel_size
    if num_groups > num_hidden_layers:
        num_groups = num_hidden_layers

    from vllm.distributed.utils import get_pp_indices

    stage_ranges = [
        get_pp_indices(num_hidden_layers, stage, pipeline_parallel_size)
        for stage in range(pipeline_parallel_size)
    ]
    stage_lengths = [end - start for start, end in stage_ranges]
    if any(stage_length <= 0 for stage_length in stage_lengths):
        raise ValueError(
            "layered prefill requires every PP stage to own at least one layer"
        )
    groups_per_stage = [1] * pipeline_parallel_size
    next_stage = 0
    for _ in range(num_groups - pipeline_parallel_size):
        for _ in range(pipeline_parallel_size):
            if groups_per_stage[next_stage] < stage_lengths[next_stage]:
                groups_per_stage[next_stage] += 1
                next_stage = (next_stage + 1) % pipeline_parallel_size
                break
            next_stage = (next_stage + 1) % pipeline_parallel_size
        else:
            raise ValueError("cannot fit layered groups into PP stage partitions")

    ranges: list[LayerGroupRange] = []
    group_id = 0
    for (stage_start, stage_end), stage_groups in zip(
        stage_ranges, groups_per_stage
    ):
        stage_ranges_for_groups = make_layer_group_ranges(
            stage_end - stage_start, stage_groups
        )
        for local_range in stage_ranges_for_groups:
            ranges.append(
                LayerGroupRange(
                    group_id,
                    stage_start + local_range.start,
                    stage_start + local_range.end,
                )
            )
            group_id += 1
    assert len(ranges) == num_groups
    return tuple(ranges)


def select_num_groups(
    prompt_tokens: int,
    num_hidden_layers: int,
    *,
    group_token_target: int = DEFAULT_GROUP_TOKEN_TARGET,
    allowed_num_groups: tuple[int, ...] = DEFAULT_ALLOWED_NUM_GROUPS,
) -> int:
    """Choose the smallest pre-captured layout not below the target bucket."""

    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be > 0")
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be > 0")
    if group_token_target <= 0:
        raise ValueError("group_token_target must be > 0")
    target = max(1, ceil(prompt_tokens / group_token_target))
    candidates = sorted(
        {
            max(1, min(int(groups), num_hidden_layers))
            for groups in allowed_num_groups
        }
    )
    if not candidates:
        candidates = [1]
    index = bisect_left(candidates, target)
    return candidates[min(index, len(candidates) - 1)]


@dataclass(frozen=True)
class LayeredPrefillPlan:
    """One scheduler step of a layered prefill cohort.

    ``query_tokens`` describe the actual prompt rows sent to the model.  They
    are intentionally independent from ``commit_tokens``: intermediate groups
    execute query rows but commit zero logical tokens.
    """

    version: int
    cohort_id: int
    group_id: int
    num_groups: int
    group_start: int
    group_end: int
    prefill_req_ids: tuple[str, ...]
    query_tokens: Mapping[str, int]
    commit_tokens: Mapping[str, int]
    reuse_kv_blocks: bool = True
    is_final_chunk: bool = True

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported layered prefill plan version {self.version}")
        if self.cohort_id < 0:
            raise ValueError("cohort_id must be non-negative")
        if self.num_groups <= 0 or not 0 <= self.group_id < self.num_groups:
            raise ValueError("invalid group_id/num_groups")
        if self.group_start < 0 or self.group_end <= self.group_start:
            raise ValueError("invalid group range")
        req_ids = tuple(self.prefill_req_ids)
        query_tokens = {
            req_id: int(value) for req_id, value in self.query_tokens.items()
        }
        commit_tokens = {
            req_id: int(value) for req_id, value in self.commit_tokens.items()
        }
        object.__setattr__(self, "prefill_req_ids", req_ids)
        object.__setattr__(self, "query_tokens", query_tokens)
        object.__setattr__(self, "commit_tokens", commit_tokens)
        if not req_ids:
            raise ValueError("a layered prefill plan must contain a request")
        if len(req_ids) != len(set(req_ids)):
            raise ValueError("prefill_req_ids must be unique")
        if set(self.query_tokens) != set(req_ids):
            raise ValueError("query_tokens must contain exactly all prefill requests")
        if set(self.commit_tokens) != set(req_ids):
            raise ValueError("commit_tokens must contain exactly all prefill requests")
        for req_id in req_ids:
            query_len = query_tokens[req_id]
            commit_len = commit_tokens[req_id]
            if query_len <= 0:
                raise ValueError("query_tokens must be positive")
            if commit_len not in (0, query_len):
                raise ValueError(
                    "commit_tokens must be zero or equal to query_tokens; "
                    "an intermediate group cannot partially commit"
                )
        if self.is_final_group and any(
            commit_tokens[req_id] != query_tokens[req_id]
            for req_id in req_ids
        ):
            raise ValueError("the final group must commit all query tokens")
        if not self.is_final_group and any(self.commit_tokens.values()):
            raise ValueError("an intermediate group cannot commit logical tokens")

    @property
    def is_final_group(self) -> bool:
        return self.group_id + 1 == self.num_groups

    @property
    def is_sampling_step(self) -> bool:
        """Only the final group of the final prompt chunk produces logits."""
        return self.is_final_group and self.is_final_chunk

    @property
    def layer_range(self) -> LayerGroupRange:
        return LayerGroupRange(self.group_id, self.group_start, self.group_end)


@dataclass
class LayeredFrontier:
    """Worker-local activation frontier for one request."""

    req_id: str
    group_id: int
    query_len: int
    hidden_states: Any
    residual: Any | None = None


@dataclass
class LayeredPrefillStateStore:
    """Small worker-local store keyed by stable request IDs, never batch rows."""

    by_req_id: dict[str, LayeredFrontier] = field(default_factory=dict)

    def get(self, req_id: str) -> LayeredFrontier | None:
        return self.by_req_id.get(req_id)

    def put(self, frontier: LayeredFrontier) -> None:
        self.by_req_id[frontier.req_id] = frontier

    def pop(self, req_id: str) -> LayeredFrontier | None:
        return self.by_req_id.pop(req_id, None)

    def clear(self, req_id: str) -> None:
        self.by_req_id.pop(req_id, None)

    def clear_many(self, req_ids: set[str] | tuple[str, ...] | list[str]) -> None:
        for req_id in req_ids:
            self.clear(req_id)

    def clear_all(self) -> None:
        self.by_req_id.clear()


@dataclass
class LayeredForwardOutput:
    """Model-side result of executing one layer group."""

    hidden_states: Any
    residual: Any | None
    is_final_layer: bool


class LayeredPrefillPolicy:
    """Pure policy/state helpers used by the scheduler integration."""

    def __init__(self, vllm_config: Any):
        self.config = LayeredPrefillConfig.from_vllm_config(vllm_config)
        self.num_hidden_layers = (
            get_num_hidden_layers(vllm_config) if self.config.enabled else 0
        )
        parallel_config = getattr(vllm_config, "parallel_config", None)
        self.pipeline_parallel_size = int(
            getattr(parallel_config, "pipeline_parallel_size", 1)
        )
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        self.max_num_batched_tokens = int(
            getattr(scheduler_config, "max_num_batched_tokens", 0) or 0
        )
        self._next_cohort_id = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def new_cohort_id(self) -> int:
        cohort_id = self._next_cohort_id
        self._next_cohort_id += 1
        return cohort_id

    def initialize_request(self, request: Any) -> None:
        if not self.enabled:
            return
        if getattr(request, "layered_prefill_enabled", False):
            return
        request.layered_prefill_enabled = True
        request.layered_prefill_cohort_id = self.new_cohort_id()
        request.layered_prefill_kv_reserved = False
        self.plan_chunk(request, request.num_computed_tokens)

    def plan_chunk(self, request: Any, chunk_start: int) -> None:
        """Plan the token chunk and layer-group layout from chunk_start.

        A prompt longer than the per-step token budget is prefilled in
        max_num_batched_tokens chunks, and every chunk is executed as
        its own layer-group sequence.  Longer chunks therefore still
        receive a finer layer-granularity split.
        """

        remaining = request.num_prompt_tokens - chunk_start
        if remaining <= 0:
            raise ValueError(
                "layered prefill chunk start must leave prompt tokens"
            )
        query_tokens = remaining
        if self.max_num_batched_tokens > 0:
            query_tokens = min(query_tokens, self.max_num_batched_tokens)
        num_groups = select_num_groups(
            query_tokens,
            self.num_hidden_layers,
            group_token_target=self.config.group_token_target,
            allowed_num_groups=self.config.allowed_num_groups,
        )
        if self.pipeline_parallel_size > 1:
            # A group must have one unambiguous PP owner.  Prefer the
            # configured layout, but never allow fewer groups than stages.
            num_groups = min(
                self.num_hidden_layers,
                max(self.pipeline_parallel_size, num_groups),
            )
        request.layered_prefill_group_id = 0
        request.layered_prefill_num_groups = num_groups
        # Keep the logical query identical to regular chunked prefill.  The
        # worker pads the physical batch for sequence-sharded execution,
        # while scheduler bookkeeping and KV commit cover the whole chunk.
        request.layered_prefill_query_tokens = query_tokens

    def make_plan(self, request: Any) -> LayeredPrefillPlan:
        if not getattr(request, "layered_prefill_enabled", False):
            raise ValueError("request is not enabled for layered prefill")
        num_groups = request.layered_prefill_num_groups
        group_id = request.layered_prefill_group_id
        ranges = make_pp_aligned_layer_group_ranges(
            self.num_hidden_layers,
            num_groups,
            self.pipeline_parallel_size,
        )
        layer_range = ranges[group_id]
        query_tokens = request.layered_prefill_query_tokens
        return LayeredPrefillPlan(
            version=1,
            cohort_id=request.layered_prefill_cohort_id,
            group_id=group_id,
            num_groups=num_groups,
            group_start=layer_range.start,
            group_end=layer_range.end,
            prefill_req_ids=(request.request_id,),
            query_tokens={request.request_id: query_tokens},
            commit_tokens={
                request.request_id: query_tokens
                if group_id + 1 == num_groups
                else 0
            },
            # The admission step sets ``layered_prefill_kv_reserved`` before
            # constructing the plan.  Group 0 still owns the initial
            # allocation; only later groups are true KV-reuse steps.
            reuse_kv_blocks=(
                request.layered_prefill_kv_reserved and group_id > 0
            ),
            is_final_chunk=(
                request.num_computed_tokens + query_tokens
                >= request.num_prompt_tokens
            ),
        )


def reset_layered_prefill_request(request: Any) -> None:
    """Reset all partial state after preemption, abort, or request finish."""

    request.layered_prefill_enabled = False
    request.layered_prefill_cohort_id = -1
    request.layered_prefill_group_id = 0
    request.layered_prefill_num_groups = 0
    request.layered_prefill_query_tokens = 0
    request.layered_prefill_kv_reserved = False
