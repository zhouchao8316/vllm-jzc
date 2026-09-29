# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

import hashlib
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    with outputs that might be needed in a subsequent (decode) step.
    Returns None if no sampled outputs are needed in the requests' next step."""

    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    max_seq_len = input_batch.max_seq_len_np
    assert max_seq_len is not None  # always populated under PP
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    # Exclude requests that we know are finished.
    not_finishing = np.maximum(old_computed, prefill_len) + 1 < max_seq_len
    need_sampled_mask = produces_sample & not_finishing
    return need_sampled_mask if need_sampled_mask.any() else None


@dataclass(frozen=True)
class RingStepPlan:
    """One sampled-token collective, identical on every PP rank.

    Issued by the scheduler with the step. ``collective_required`` is the
    only participation bit. A local mask may decide whether a received row
    is written; it must not decide whether this rank enters the collective.
    """

    step_id: int
    collective_required: bool
    payload_rows: int
    sampled_request_order: tuple[str, ...]
    sample_width: int
    plan_hash: str

    def __post_init__(self) -> None:
        if self.payload_rows != len(self.sampled_request_order):
            raise ValueError(
                "RingStepPlan payload_rows does not match request order: "
                f"rows={self.payload_rows} "
                f"order={len(self.sampled_request_order)}"
            )
        if self.sample_width < 1:
            raise ValueError(f"RingStepPlan sample_width must be positive, got {self.sample_width}")


@dataclass(frozen=True)
class RingStepRecord:
    """What one rank did for one sample step. Ranks must reconcile."""

    step_id: int
    role: str
    participated: bool
    payload_rows: int
    sample_width: int
    request_order: tuple[str, ...]
    plan_hash: str
    apply_count: int


def ring_plan_hash(
    *,
    step_id: int,
    collective_required: bool,
    sampled_request_order: Sequence[str],
    sample_width: int,
) -> str:
    blob = (
        f"{int(step_id)}|{int(collective_required)}|"
        f"{len(sampled_request_order)}|{int(sample_width)}|"
        f"{','.join(sampled_request_order)}"
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def build_ring_step_plan(
    *,
    step_id: int,
    pp_size: int,
    sample_width: int,
    request_order: Sequence[str],
    old_computed: dict[str, int],
    num_scheduled: dict[str, int],
    prefill_len: dict[str, int],
    max_seq_len: dict[str, int],
    sampling_step: bool | None,
) -> RingStepPlan | None:
    """Build the plan the scheduler broadcasts.

    ``sampling_step is False`` is the layered global "no sample" decision
    (intermediate group, or a non-final chunk). Every rank skips, including
    ranks whose local cursor would treat the chunk as a finished prefill.
    ``None`` means a regular token-chunk step: participation follows the
    same mask as ``compute_need_sampled_mask``, computed once here.
    """
    if pp_size <= 1:
        return None
    order = tuple(request_order)
    missing = [
        req_id
        for req_id in order
        if req_id not in old_computed
        or req_id not in num_scheduled
        or req_id not in prefill_len
        or req_id not in max_seq_len
    ]
    if missing:
        raise RuntimeError(
            f"RingStepPlan is missing scheduler fields for {missing}"
        )
    if sampling_step is False or not order:
        collective_required = False
    else:
        # Same predicate as the worker mask, evaluated once on the scheduler.
        batch = SimpleNamespace(
            num_computed_tokens_np=np.array(
                [old_computed[req_id] for req_id in order], dtype=np.int64
            ),
            prefill_len_np=np.array(
                [prefill_len[req_id] for req_id in order], dtype=np.int64
            ),
            max_seq_len_np=np.array(
                [max_seq_len[req_id] for req_id in order], dtype=np.int64
            ),
            num_scheduled_tokens=np.array(
                [num_scheduled[req_id] for req_id in order], dtype=np.int64
            ),
        )
        collective_required = compute_need_sampled_mask(batch) is not None
    return RingStepPlan(
        step_id=int(step_id),
        collective_required=collective_required,
        payload_rows=len(order),
        sampled_request_order=order,
        sample_width=int(sample_width),
        plan_hash=ring_plan_hash(
            step_id=int(step_id),
            collective_required=collective_required,
            sampled_request_order=order,
            sample_width=int(sample_width),
        ),
    )


def ring_collective_action(plan: RingStepPlan | None) -> str:
    """Return ``enter``, ``skip``, or ``missing``.

    Local masks are not an input. ``missing`` is a protocol error: a PP
    sample step must not invent its own skip.
    """
    if plan is None:
        return "missing"
    return "enter" if plan.collective_required else "skip"


def align_sampled_payload(
    plan: RingStepPlan,
    req_ids: Sequence[str],
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder local sample rows into ``plan.sampled_request_order``.

    Rows the plan does not name are dropped (a cancelled rider that this
    rank still captured). Plan rows missing locally are an empty sample
    (token ``-1``, ``num_sampled`` 0), not a reason to shrink the payload.
    """
    if sampled_token_ids.ndim == 1:
        sampled_token_ids = sampled_token_ids.unsqueeze(1)
    width = int(plan.sample_width)
    rows = int(plan.payload_rows)
    tokens = sampled_token_ids.new_full((rows, width), -1)
    sampled = num_sampled.new_zeros(rows)
    rejected = num_rejected.new_zeros(rows)
    local_index = {req_id: index for index, req_id in enumerate(req_ids)}
    n_local = int(sampled_token_ids.shape[0])
    for row, req_id in enumerate(plan.sampled_request_order):
        index = local_index.get(req_id)
        if index is None or index >= n_local:
            continue
        src = sampled_token_ids[index]
        if src.ndim == 0:
            src = src.reshape(1)
        take = min(width, int(src.shape[-1]))
        tokens[row, :take] = src[:take]
        sampled[row] = num_sampled[index]
        rejected[row] = num_rejected[index]
    return tokens, sampled, rejected


def local_apply_mask(
    plan: RingStepPlan,
    req_ids: Sequence[str],
    local_need: np.ndarray | None,
) -> np.ndarray:
    """Bool mask of length ``payload_rows``.

    True only for plan rows that this rank both has and locally marked as
    needing a later decode write. Never consulted for participation.
    """
    mask = np.zeros(plan.payload_rows, dtype=bool)
    if local_need is None:
        return mask
    local_index = {req_id: index for index, req_id in enumerate(req_ids)}
    for row, req_id in enumerate(plan.sampled_request_order):
        index = local_index.get(req_id)
        if index is None or index >= len(local_need):
            continue
        mask[row] = bool(local_need[index])
    return mask


def reconcile_ring_records(records: Sequence[RingStepRecord]) -> None:
    """Fail if ranks disagree on participation, shape, or request order."""
    grouped: dict[int, list[RingStepRecord]] = {}
    for record in records:
        grouped.setdefault(record.step_id, []).append(record)
    for step_id, group in grouped.items():
        hashes = {record.plan_hash for record in group}
        participated = {record.participated for record in group}
        shapes = {(record.payload_rows, record.sample_width) for record in group}
        orders = {record.request_order for record in group}
        if len(hashes) != 1 or len(participated) != 1 or len(shapes) != 1 or len(orders) != 1:
            raise AssertionError(
                f"Ring step {step_id} diverged across ranks: "
                f"hash={hashes} participated={participated} "
                f"shape={shapes} order={orders}"
            )
        if len(group) < 2:
            raise AssertionError(
                f"Ring step {step_id} has only {len(group)} rank record"
            )


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.
    """

    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ):
        self.is_last_rank = get_pp_group().is_last_rank
        self.last_rank = get_pp_group().last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * get_pp_group().world_size)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)
        # Shared plan for the step currently being sampled. Participation
        # follows this object; it is never recomputed from a local mask.
        self.ring_step_plan: RingStepPlan | None = None
        self.ring_ledger: list[RingStepRecord] = []

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def bind_plan(self, plan: RingStepPlan | None) -> None:
        self.ring_step_plan = plan

    def _record_ring(
        self, *, role: str, participated: bool, apply_count: int
    ) -> RingStepRecord:
        plan = self.ring_step_plan
        if plan is None:
            raise RuntimeError(
                "PP sampled-token collective has no RingStepPlan; "
                "refusing to decide participation from the local mask"
            )
        record = RingStepRecord(
            step_id=plan.step_id,
            role=role,
            participated=participated,
            payload_rows=plan.payload_rows,
            sample_width=plan.sample_width,
            request_order=plan.sampled_request_order,
            plan_hash=plan.plan_hash,
            apply_count=apply_count,
        )
        self.ring_ledger.append(record)
        if len(self.ring_ledger) > 256:
            del self.ring_ledger[:-256]
        return record

    def record_ring_skip(self, role: str) -> None:
        """Flush skipped the collective. Do not also enter broadcast/receive."""
        if ring_collective_action(self.ring_step_plan) != "skip":
            raise RuntimeError(
                "record_ring_skip requires a plan that globally skips"
            )
        self._record_ring(role=role, participated=False, apply_count=0)

    def _require_action(self, role: str) -> str:
        action = ring_collective_action(self.ring_step_plan)
        if action == "missing":
            raise RuntimeError(
                "PP sampled-token collective has no RingStepPlan; "
                "refusing to decide participation from the local mask"
            )
        if action == "skip":
            self._record_ring(role=role, participated=False, apply_count=0)
        return action

    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        if slot is None:
            return None

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        self.main_stream.wait_event(slot.event)
        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
        )

    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff every received row should be applied locally.

        Entering the collective follows ``ring_step_plan`` only. The local
        mask is stored on the queue slot and applied at consume time.
        """
        assert not self.is_last_rank
        action = self._require_action("receive")
        if action == "skip":
            # Leave this step's reserved slot as None. Every rank skipped.
            return False

        plan = self.ring_step_plan
        assert plan is not None
        req_ids = list(input_batch.req_ids)[: input_batch.num_reqs]
        apply_mask = local_apply_mask(
            plan, req_ids, compute_need_sampled_mask(input_batch)
        )
        self._record_ring(
            role="receive",
            participated=True,
            apply_count=int(apply_mask.sum()),
        )
        self._enter_receive(input_batch, plan, apply_mask)
        return bool(apply_mask.size and apply_mask.all())

    def _enter_receive(
        self,
        input_batch: InputBatch,
        plan: RingStepPlan,
        apply_mask: np.ndarray,
    ) -> None:
        req_ids = list(input_batch.req_ids)[: input_batch.num_reqs]
        local_index = {req_id: index for index, req_id in enumerate(req_ids)}
        idx_np = np.full(plan.payload_rows, -1, dtype=np.int32)
        for row, req_id in enumerate(plan.sampled_request_order):
            index = local_index.get(req_id)
            if index is None:
                continue
            idx_np[row] = int(input_batch.idx_mapping_np[index])
        safe_idx = np.where(idx_np < 0, 0, idx_np)
        gen_at_receive_np = self.req_idx_gen_np[safe_idx]
        idx_mapping = async_copy_to_gpu(idx_np, device=self.device)

        num_reqs = plan.payload_rows
        sample_width = plan.sample_width
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, sample_width, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            idx_mapping,
            idx_np,
            apply_mask,
            gen_at_receive_np,
        )

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        action = self._require_action("broadcast")
        if action == "skip":
            return

        plan = self.ring_step_plan
        assert plan is not None
        assert sampled_token_ids.dtype == torch.int64
        req_ids = list(input_batch.req_ids)[: input_batch.num_reqs]
        if num_sampled is None:
            num_sampled = torch.zeros(
                len(req_ids), dtype=torch.int32, device=sampled_token_ids.device
            )
        if num_rejected is None:
            num_rejected = torch.zeros(
                len(req_ids), dtype=torch.int32, device=sampled_token_ids.device
            )
        tokens, sampled, rejected = align_sampled_payload(
            plan,
            req_ids,
            sampled_token_ids,
            num_sampled,
            num_rejected,
        )
        self._record_ring(
            role="broadcast",
            participated=True,
            apply_count=int(sampled.ne(0).sum().item()),
        )
        self._enter_broadcast(tokens, sampled, rejected)

    def _enter_broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                sampled_token_ids.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
