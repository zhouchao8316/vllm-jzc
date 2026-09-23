# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.core.layered_prefill import (
    LayeredFrontier,
    LayeredPrefillConfig,
    LayeredPrefillPlan,
    LayeredPrefillPolicy,
    LayeredPrefillStateStore,
    make_layer_group_ranges,
    make_pp_aligned_layer_group_ranges,
    select_num_groups,
)


def _config(
    *,
    enabled: bool = True,
    layers: int = 8,
    tensor_parallel_size: int = 1,
    use_sequence_parallel_moe: bool = False,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            num_hidden_layers=layers,
            enforce_eager=True,
        ),
        additional_config={
            "enable_dsa_cp": False,
            "scheduler_config": {
                "layered_prefill_config": {"enabled": enabled}
            }
        },
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=1,
            use_sequence_parallel_moe=use_sequence_parallel_moe,
        ),
    )


def test_layer_group_ranges_cover_each_layer_once():
    ranges = make_layer_group_ranges(10, 4)

    assert [(item.start, item.end) for item in ranges] == [
        (0, 3),
        (3, 6),
        (6, 8),
        (8, 10),
    ]
    assert ranges[-1].end == 10
    assert sum(item.end - item.start for item in ranges) == 10


def test_pp_layer_group_ranges_do_not_cross_stage_boundaries():
    ranges = make_pp_aligned_layer_group_ranges(8, 4, 2)

    assert [(item.start, item.end) for item in ranges] == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
    ]


def test_pp_layer_group_ranges_follow_custom_partition(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "2,6")
    ranges = make_pp_aligned_layer_group_ranges(8, 4, 2)

    assert [(item.start, item.end) for item in ranges] == [
        (0, 1),
        (1, 2),
        (2, 5),
        (5, 8),
    ]


@pytest.mark.parametrize(
    "prompt_tokens, expected_groups",
    [(1, 1), (512, 1), (513, 2), (2048, 4), (10000, 16)],
)
def test_select_num_groups_uses_stable_layout_buckets(prompt_tokens, expected_groups):
    assert select_num_groups(prompt_tokens, 32) == expected_groups


def test_plan_separates_query_and_logical_commit():
    plan = LayeredPrefillPlan(
        version=1,
        cohort_id=3,
        group_id=0,
        num_groups=2,
        group_start=0,
        group_end=4,
        prefill_req_ids=("req",),
        query_tokens={"req": 16},
        commit_tokens={"req": 0},
    )

    assert plan.is_final_group is False
    assert plan.is_sampling_step is False
    assert plan.query_tokens["req"] == 16
    assert plan.commit_tokens["req"] == 0

    final_plan = LayeredPrefillPlan(
        version=1,
        cohort_id=3,
        group_id=1,
        num_groups=2,
        group_start=4,
        group_end=8,
        prefill_req_ids=("req",),
        query_tokens={"req": 16},
        commit_tokens={"req": 16},
    )
    assert final_plan.is_final_group is True
    assert final_plan.is_final_chunk is True
    assert final_plan.is_sampling_step is True

    last_group_non_final_chunk = LayeredPrefillPlan(
        version=1,
        cohort_id=3,
        group_id=1,
        num_groups=2,
        group_start=4,
        group_end=8,
        prefill_req_ids=("req",),
        query_tokens={"req": 16},
        commit_tokens={"req": 16},
        is_final_chunk=False,
    )
    assert last_group_non_final_chunk.is_final_group is True
    assert last_group_non_final_chunk.is_sampling_step is False


def test_intermediate_plan_rejects_partial_commit():
    with pytest.raises(ValueError, match="intermediate group"):
        LayeredPrefillPlan(
            version=1,
            cohort_id=0,
            group_id=0,
            num_groups=2,
            group_start=0,
            group_end=1,
            prefill_req_ids=("req",),
            query_tokens={"req": 2},
            commit_tokens={"req": 1},
        )


def test_policy_initializes_request_and_advances_groups():
    policy = LayeredPrefillPolicy(_config(layers=8))
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=513,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
    )

    policy.initialize_request(request)
    assert request.layered_prefill_num_groups == 2
    assert request.layered_prefill_group_id == 0

    first = policy.make_plan(request)
    assert first.group_id == 0
    assert first.commit_tokens["req"] == 0
    assert first.is_sampling_step is False
    assert first.reuse_kv_blocks is False
    assert first.cached_tokens["req"] == 0
    assert request.layered_prefill_cached_tokens == 0
    request.layered_prefill_group_id += 1
    request.layered_prefill_kv_reserved = True
    second = policy.make_plan(request)
    assert second.group_id == 1
    assert second.commit_tokens["req"] == 513
    assert second.reuse_kv_blocks is True
    assert second.is_sampling_step is True


def test_plan_carries_prefix_cached_tokens():
    policy = LayeredPrefillPolicy(_config(layers=8))
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=64,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
        layered_prefill_cached_tokens=0,
    )
    policy.initialize_request(request)
    request.layered_prefill_cached_tokens = 16
    request.layered_prefill_query_tokens = 48
    plan = policy.make_plan(request)
    assert plan.query_tokens["req"] == 48
    assert plan.cached_tokens["req"] == 16


def test_reset_clears_cached_tokens():
    from vllm.v1.core.layered_prefill import reset_layered_prefill_request

    request = SimpleNamespace(
        layered_prefill_enabled=True,
        layered_prefill_cohort_id=3,
        layered_prefill_group_id=1,
        layered_prefill_num_groups=2,
        layered_prefill_query_tokens=48,
        layered_prefill_kv_reserved=True,
        layered_prefill_cached_tokens=16,
    )
    reset_layered_prefill_request(request)
    assert request.layered_prefill_enabled is False
    assert request.layered_prefill_cached_tokens == 0
    assert request.layered_prefill_query_tokens == 0


def test_eligibility_allows_prefix_hit_on_layered_request():
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    sampling = SimpleNamespace(
        logprobs=None, prompt_logprobs=None, logprob_token_ids=None
    )
    waiting = SimpleNamespace(
        status=RequestStatus.WAITING,
        pooling_params=None,
        num_prompt_tokens=64,
        num_computed_tokens=0,
        output_token_ids=[],
        sampling_params=sampling,
        use_structured_output=False,
        layered_prefill_enabled=False,
    )
    assert Scheduler._is_layered_request_eligible(waiting)

    running_hit = SimpleNamespace(
        status=RequestStatus.RUNNING,
        pooling_params=None,
        num_prompt_tokens=64,
        num_computed_tokens=16,
        output_token_ids=[],
        sampling_params=sampling,
        use_structured_output=False,
        layered_prefill_enabled=True,
    )
    assert Scheduler._is_layered_request_eligible(running_hit)

    waiting_hit_not_layered = SimpleNamespace(
        status=RequestStatus.WAITING,
        pooling_params=None,
        num_prompt_tokens=64,
        num_computed_tokens=16,
        output_token_ids=[],
        sampling_params=sampling,
        use_structured_output=False,
        layered_prefill_enabled=False,
    )
    assert not Scheduler._is_layered_request_eligible(waiting_hit_not_layered)


def test_policy_aligns_pp_groups_with_stage_partitions():
    config = _config(layers=8)
    config.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=1,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
    )

    policy.initialize_request(request)
    assert request.layered_prefill_num_groups == 2
    plan = policy.make_plan(request)
    assert (plan.group_start, plan.group_end) == (0, 4)


def test_policy_keeps_unaligned_tail_in_layered_query():
    config = _config(layers=8, tensor_parallel_size=8)
    config.model_config.hf_text_config = SimpleNamespace(index_topk=2048)
    config.additional_config["enable_dsa_cp"] = True
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=65540,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
    )

    policy.initialize_request(request)
    assert request.layered_prefill_query_tokens == 65540

    plan = policy.make_plan(request)
    assert plan.query_tokens["req"] == 65540


def test_plan_chunk_only_samples_on_last_chunk():
    config = _config(layers=8)
    config.scheduler_config = SimpleNamespace(max_num_batched_tokens=512)
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=1024,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
        layered_prefill_cached_tokens=0,
    )

    policy.initialize_request(request)
    assert request.layered_prefill_query_tokens == 512
    first = policy.make_plan(request)
    assert first.is_final_chunk is False
    assert first.is_sampling_step is False

    request.layered_prefill_group_id = request.layered_prefill_num_groups - 1
    last_of_first_chunk = policy.make_plan(request)
    assert last_of_first_chunk.is_final_group is True
    assert last_of_first_chunk.is_sampling_step is False

    request.num_computed_tokens = 512
    policy.plan_chunk(request, 512)
    last_chunk = policy.make_plan(request)
    assert last_chunk.is_final_chunk is True
    request.layered_prefill_group_id = request.layered_prefill_num_groups - 1
    last = policy.make_plan(request)
    assert last.is_final_chunk is True
    assert last.is_sampling_step is True


def test_plan_chunk_caps_query_after_prefix_hit():
    config = _config(layers=8)
    config.scheduler_config = SimpleNamespace(max_num_batched_tokens=512)
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=1024,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
        layered_prefill_cached_tokens=0,
    )
    policy.initialize_request(request)
    policy.plan_chunk(request, 16)
    assert request.layered_prefill_cached_tokens == 16
    assert request.layered_prefill_query_tokens == 512
    plan = policy.make_plan(request)
    assert plan.cached_tokens["req"] == 16
    assert plan.is_final_chunk is False


def test_v2_prefix_hit_uses_plan_chunk_not_full_remainder():
    from vllm.v1.core.sched.scheduler import Scheduler

    config = _config(layers=8)
    config.scheduler_config = SimpleNamespace(max_num_batched_tokens=512)
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=1024,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
        layered_prefill_cached_tokens=0,
    )
    policy.initialize_request(request)
    sched = Scheduler.__new__(Scheduler)
    sched.use_v2_model_runner = True
    sched.layered_prefill_policy = policy
    sched._apply_layered_prefix_hit(request, 16)
    assert request.layered_prefill_query_tokens == 512
    assert request.layered_prefill_cached_tokens == 16


def test_v1_prefix_hit_keeps_full_remainder():
    from vllm.v1.core.sched.scheduler import Scheduler

    config = _config(layers=8)
    config.scheduler_config = SimpleNamespace(max_num_batched_tokens=512)
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=1024,
        num_computed_tokens=0,
        layered_prefill_enabled=False,
        layered_prefill_group_id=0,
        layered_prefill_num_groups=0,
        layered_prefill_query_tokens=0,
        layered_prefill_cohort_id=-1,
        layered_prefill_kv_reserved=False,
        layered_prefill_cached_tokens=0,
    )
    policy.initialize_request(request)
    sched = Scheduler.__new__(Scheduler)
    sched.use_v2_model_runner = False
    sched.layered_prefill_policy = policy
    sched._apply_layered_prefix_hit(request, 16)
    assert request.layered_prefill_query_tokens == 1008
    assert request.layered_prefill_cached_tokens == 16


def test_frontier_store_is_keyed_by_request_id():
    store = LayeredPrefillStateStore()
    frontier = LayeredFrontier(
        req_id="req",
        group_id=1,
        query_len=2,
        hidden_states="hidden",
        residual="residual",
    )
    store.put(frontier)
    assert store.get("req") is frontier
    assert store.pop("req") is frontier
    assert store.get("req") is None


def test_disabled_config_does_not_require_model_layer_count():
    config = SimpleNamespace(
        model_config=SimpleNamespace(),
        additional_config={"scheduler_config": {}},
    )
    policy = LayeredPrefillPolicy(config)
    assert policy.enabled is False
    assert policy.num_hidden_layers == 0
    assert LayeredPrefillConfig().enabled is False


def test_config_parses_phase_one_scheduler_options():
    config = LayeredPrefillConfig.from_vllm_config(
        SimpleNamespace(
            additional_config={
                "scheduler_config": {
                    "layered_prefill_config": {
                        "enabled": True,
                        "group_token_target": 1024,
                        "allowed_num_groups": [1, 2, 4],
                        "max_groups_per_step": 1,
                    }
                }
            }
        )
    )

    assert config.enabled is True
    assert config.group_token_target == 1024
    assert config.allowed_num_groups == (1, 2, 4)
    assert config.fuse_mixed_batch is False


def test_config_parses_fuse_mixed_batch():
    config = LayeredPrefillConfig.from_vllm_config(
        SimpleNamespace(
            additional_config={
                "scheduler_config": {
                    "layered_prefill_config": {
                        "enabled": True,
                        "fuse_mixed_batch": True,
                    }
                }
            }
        )
    )
    assert config.fuse_mixed_batch is True
    assert LayeredPrefillConfig().fuse_mixed_batch is False


def test_config_parses_same_layer_batch():
    config = LayeredPrefillConfig.from_vllm_config(
        SimpleNamespace(
            additional_config={
                "scheduler_config": {
                    "layered_prefill_config": {
                        "enabled": True,
                        "same_layer_batch": True,
                        "p_group_decode_budget_ms": 80,
                    }
                }
            }
        )
    )
    assert config.same_layer_batch is True
    assert config.p_group_decode_budget_ms == 80.0
    assert LayeredPrefillConfig().same_layer_batch is False


def test_config_rejects_fuse_and_same_layer_together():
    with pytest.raises(ValueError, match="mutually exclusive"):
        LayeredPrefillConfig(enabled=True, fuse_mixed_batch=True, same_layer_batch=True)


def test_config_rejects_multiple_groups_per_step_in_phase_one():
    with pytest.raises(ValueError, match="max_groups_per_step=1"):
        LayeredPrefillConfig(enabled=True, max_groups_per_step=2)


def test_merge_consecutive_groups_on_pp1_is_a_wider_layer_range():
    from vllm.v1.core.layered_prefill import merge_consecutive_layer_groups

    ranges = make_layer_group_ranges(8, 4)
    merged = merge_consecutive_layer_groups(ranges, 0, 2)
    assert (merged.start, merged.end) == (0, 4)
    assert merged.consumed == 2
    assert merged.includes_final is False

    tail = merge_consecutive_layer_groups(ranges, 2, 4)
    assert (tail.start, tail.end) == (4, 8)
    assert tail.consumed == 2
    assert tail.includes_final is True
    assert tail.group_id == 2


def test_merge_consecutive_groups_does_not_cross_pp_stage():
    from vllm.v1.core.layered_prefill import merge_consecutive_layer_groups

    ranges = make_pp_aligned_layer_group_ranges(8, 4, 2)
    # [(0,2), (2,4), (4,6), (6,8)] — stage boundary at 4.
    same_stage = merge_consecutive_layer_groups(
        ranges, 0, 4, pipeline_parallel_size=2, num_hidden_layers=8
    )
    assert (same_stage.start, same_stage.end) == (0, 4)
    assert same_stage.consumed == 2

    across = merge_consecutive_layer_groups(
        ranges, 1, 2, pipeline_parallel_size=2, num_hidden_layers=8
    )
    assert (across.start, across.end) == (2, 4)
    assert across.consumed == 1


def test_mixed_compaction_visits_p_only_in_active_groups():
    import numpy as np

    rng = np.random.default_rng(0)
    n_d, n_p, hidden, layers = 2, 5, 8, 6
    gs, ge = 2, 4
    weights = [rng.normal(size=(hidden, hidden)).astype(np.float64) for _ in range(layers)]
    d_in = rng.normal(size=(n_d, hidden)).astype(np.float64)
    p_embed = rng.normal(size=(n_p, hidden)).astype(np.float64)

    def apply(h, start, end):
        for i in range(start, end):
            h = h @ weights[i]
        return h

    p_frontier = apply(p_embed.copy(), 0, gs)
    two_call_d = apply(d_in.copy(), 0, layers)
    two_call_p = apply(p_frontier.copy(), gs, ge)

    h = np.concatenate([d_in.copy(), p_frontier.copy()], axis=0)
    p_visits = np.zeros(layers, dtype=int)
    d_visits = np.zeros(layers, dtype=int)
    for i in range(layers):
        if gs <= i < ge:
            h = h @ weights[i]
            d_visits[i] += 1
            p_visits[i] += 1
        else:
            h[:n_d] = h[:n_d] @ weights[i]
            d_visits[i] += 1
    mixed_d, mixed_p = h[:n_d], h[n_d:]

    assert np.allclose(mixed_d, two_call_d)
    assert np.allclose(mixed_p, two_call_p)
    assert d_visits.tolist() == [1] * layers
    assert p_visits.tolist() == [1 if gs <= i < ge else 0 for i in range(layers)]

    naive = np.concatenate([d_in.copy(), p_frontier.copy()], axis=0)
    naive_p_visits = np.zeros(layers, dtype=int)
    for i in range(layers):
        naive = naive @ weights[i]
        naive_p_visits[i] += 1
    assert naive_p_visits.tolist() == [1] * layers
    assert not np.allclose(naive[n_d:], two_call_p)


def test_fused_mixed_both_visit_active_group_only():
    """Fused mixed: D and P both visit only the active layer group each step.

    After every group, both match a single full-stack apply. This is the
    intended fuse_mixed_batch GEMM pattern, not serial D-all-layers + P-group.
    """
    import numpy as np

    rng = np.random.default_rng(1)
    n_d, n_p, hidden, layers, groups = 2, 5, 8, 6, 2
    weights = [rng.normal(size=(hidden, hidden)).astype(np.float64) for _ in range(layers)]
    d_in = rng.normal(size=(n_d, hidden)).astype(np.float64)
    p_in = rng.normal(size=(n_p, hidden)).astype(np.float64)

    def apply(h, start, end):
        for i in range(start, end):
            h = h @ weights[i]
        return h

    full = apply(np.concatenate([d_in.copy(), p_in.copy()], axis=0), 0, layers)
    h = np.concatenate([d_in.copy(), p_in.copy()], axis=0)
    group_width = layers // groups
    d_visits = np.zeros(layers, dtype=int)
    p_visits = np.zeros(layers, dtype=int)
    for group_id in range(groups):
        gs, ge = group_id * group_width, (group_id + 1) * group_width
        h = apply(h, gs, ge)
        d_visits[gs:ge] += 1
        p_visits[gs:ge] += 1
    assert np.allclose(h, full)
    assert d_visits.tolist() == [1] * layers
    assert p_visits.tolist() == [1] * layers


def _enable_layered_scheduler(scheduler, **layered_cfg):
    from vllm.v1.core.layered_prefill import LayeredPrefillPolicy

    scheduler.vllm_config.model_config.enforce_eager = True
    cfg = {
        "enabled": True,
        "mode": "one_group",
        "require_eager": True,
        "require_pd_mixed": False,
        "allowed_num_groups": [2],
        "group_token_target": 1,
        "max_groups_per_step": 1,
    }
    cfg.update(layered_cfg)
    scheduler.vllm_config.additional_config = {
        "scheduler_config": {"layered_prefill_config": cfg}
    }
    scheduler.layered_prefill_policy = LayeredPrefillPolicy(scheduler.vllm_config)
    assert scheduler.layered_prefill_policy.enabled


def _local_scheduler(**kwargs):
    from tests.v1.core.utils import create_scheduler

    opts = dict(
        model="/gjc-workspace/models/Qwen3-30B-A3B",
        skip_tokenizer_init=True,
        max_num_seqs=2,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
    )
    opts.update(kwargs)
    return create_scheduler(**opts)


def _empty_runner_output(req_ids: list[str]):
    from vllm.v1.outputs import ModelRunnerOutput

    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        sampled_token_ids=[[] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _sampled_runner_output(req_id: str, token_id: int = 1000):
    from vllm.v1.outputs import ModelRunnerOutput

    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[[token_id]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_in_flight_layer_group_does_not_steal_waiting_prompt():
    """In-flight group: hold waiters; do not start their regular prefill."""
    from tests.v1.core.utils import create_requests

    scheduler = _local_scheduler()
    _enable_layered_scheduler(scheduler)
    first, second = create_requests(num_requests=2, num_tokens=8, max_tokens=8)
    scheduler.add_request(first)
    scheduler.add_request(second)

    first_g0 = scheduler.schedule()
    assert first_g0.layered_prefill_plan is not None
    assert first.request_id in first_g0.num_scheduled_tokens
    assert second.request_id not in first_g0.num_scheduled_tokens

    # Do not update_from_output: the first group is still in flight, as in PP.
    held = scheduler.schedule()
    assert held.layered_prefill_plan is None
    assert second.request_id not in held.num_scheduled_tokens
    assert first.request_id not in held.num_scheduled_tokens
    assert second.status.name in ("WAITING", "PREEMPTED")


def test_async_in_flight_holds_waiting_prompt(monkeypatch: pytest.MonkeyPatch):
    from tests.v1.core.utils import create_requests

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    scheduler = _local_scheduler(
        async_scheduling=True,
        use_v2_model_runner=True,
    )
    _enable_layered_scheduler(scheduler)
    first, second = create_requests(num_requests=2, num_tokens=8, max_tokens=8)
    scheduler.add_request(first)
    scheduler.add_request(second)

    first_g0 = scheduler.schedule()
    assert first_g0.layered_prefill_plan is not None
    assert second.request_id not in first_g0.num_scheduled_tokens

    held = scheduler.schedule()
    assert held.layered_prefill_plan is None
    assert second.request_id not in held.num_scheduled_tokens
    assert first.request_id not in held.num_scheduled_tokens


def test_first_sample_mixes_decode_with_next_layered_prompt():
    from tests.v1.core.utils import create_requests

    scheduler = _local_scheduler()
    _enable_layered_scheduler(scheduler)
    first, second = create_requests(num_requests=2, num_tokens=8, max_tokens=8)
    scheduler.add_request(first)
    scheduler.add_request(second)

    first_g0 = scheduler.schedule()
    assert first_g0.layered_prefill_plan is not None
    scheduler.update_from_output(first_g0, _empty_runner_output([first.request_id]))

    first_g1 = scheduler.schedule()
    assert first_g1.layered_prefill_plan is not None
    assert first_g1.layered_prefill_plan.is_final_group
    assert second.request_id not in first_g1.num_scheduled_tokens

    # Sample has not landed yet: hold the waiter instead of starting a P-only step.
    held = scheduler.schedule()
    assert second.request_id not in held.num_scheduled_tokens

    scheduler.update_from_output(
        first_g1, _sampled_runner_output(first.request_id, token_id=1000)
    )

    mixed = scheduler.schedule()
    assert mixed.layered_prefill_plan is not None
    assert mixed.layered_prefill_plan.group_id == 0
    assert second.request_id in mixed.num_scheduled_tokens
    assert first.request_id in mixed.num_scheduled_tokens

    from vllm.v1.outputs import ModelRunnerOutput

    mixed_ids = list(mixed.num_scheduled_tokens)
    scheduler.update_from_output(
        mixed,
        ModelRunnerOutput(
            req_ids=mixed_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(mixed_ids)},
            sampled_token_ids=[
                [1001] if req_id == first.request_id else [] for req_id in mixed_ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    final_mix = scheduler.schedule()
    assert final_mix.layered_prefill_plan is not None
    assert final_mix.layered_prefill_plan.is_final_group
    assert second.request_id in final_mix.num_scheduled_tokens
    assert first.request_id in final_mix.num_scheduled_tokens


def test_fuse_mixed_holds_decode_token_progress_until_last_group():
    """fuse_mixed_batch replays the same D token until the sampling group."""
    from tests.v1.core.utils import create_requests

    scheduler = _local_scheduler()
    _enable_layered_scheduler(scheduler, fuse_mixed_batch=True)
    first, second = create_requests(num_requests=2, num_tokens=8, max_tokens=8)
    scheduler.add_request(first)
    scheduler.add_request(second)

    first_g0 = scheduler.schedule()
    assert first_g0.layered_prefill_plan is not None
    scheduler.update_from_output(first_g0, _empty_runner_output([first.request_id]))

    first_g1 = scheduler.schedule()
    assert first_g1.layered_prefill_plan is not None
    assert first_g1.layered_prefill_plan.is_final_group
    scheduler.update_from_output(
        first_g1, _sampled_runner_output(first.request_id, token_id=1000)
    )
    prompt_cursor = first.num_computed_tokens
    assert prompt_cursor >= first.num_prompt_tokens

    mixed = scheduler.schedule()
    assert mixed.layered_prefill_plan is not None
    assert mixed.layered_prefill_plan.group_id == 0
    assert first.request_id in mixed.num_scheduled_tokens
    assert second.request_id in mixed.num_scheduled_tokens
    assert mixed.num_scheduled_tokens[first.request_id] == 1
    # Same decode token is still in flight; cursor does not advance.
    assert first.num_computed_tokens == prompt_cursor
    assert first.request_id in (second.layered_fused_decode_ids or [])

    mixed_ids = list(mixed.num_scheduled_tokens)
    scheduler.update_from_output(mixed, _empty_runner_output(mixed_ids))
    assert first.num_computed_tokens == prompt_cursor

    final_mix = scheduler.schedule()
    assert final_mix.layered_prefill_plan is not None
    assert final_mix.layered_prefill_plan.is_final_group
    assert final_mix.layered_prefill_plan.is_sampling_step
    assert first.request_id in final_mix.num_scheduled_tokens
    assert second.request_id in final_mix.num_scheduled_tokens
    # Sampling step commits the replayed decode token.
    assert first.num_computed_tokens == prompt_cursor + 1
    assert second.layered_fused_decode_ids is None
    assert not first.layered_fused_decode_slot


def _drive_layered_until_sample(scheduler, token_for):
    """Run layer groups until the sampling step, then apply sampled tokens."""
    from vllm.v1.outputs import ModelRunnerOutput

    step = scheduler.schedule()
    while (
        step.layered_prefill_plan is not None
        and not step.layered_prefill_plan.is_sampling_step
    ):
        scheduler.update_from_output(
            step, _empty_runner_output(list(step.num_scheduled_tokens))
        )
        step = scheduler.schedule()
    assert step.layered_prefill_plan is not None
    assert step.layered_prefill_plan.is_sampling_step
    req_ids = list(step.num_scheduled_tokens)
    scheduler.update_from_output(
        step,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=[[token_for(req_id)] for req_id in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return step


def test_fuse_kv_reject_keeps_ready_decodes_runnable():
    """Third P cannot get KV. Ready decodes still run, finish, and free KV."""
    from tests.v1.core.utils import create_requests

    # One block is the null block. Two live prompts fill the rest, so the
    # third admission fails until a decode request finishes and frees KV.
    scheduler = _local_scheduler(max_num_seqs=4, num_blocks=3, max_num_batched_tokens=32)
    _enable_layered_scheduler(scheduler, fuse_mixed_batch=True)
    first, second, third = create_requests(
        num_requests=3, num_tokens=8, max_tokens=4, block_size=16
    )
    scheduler.add_request(first)
    _drive_layered_until_sample(scheduler, lambda _req_id: 1000)
    assert first.num_output_tokens == 1
    assert not first.layered_fused_decode_slot

    scheduler.add_request(second)
    scheduler.add_request(third)
    _drive_layered_until_sample(scheduler, lambda _req_id: 1001)
    assert second.num_output_tokens == 1
    assert not first.layered_fused_decode_slot
    assert not second.layered_fused_decode_slot

    before = {
        first.request_id: first.num_computed_tokens,
        second.request_id: second.num_computed_tokens,
    }
    stalled = scheduler.schedule()
    assert third.request_id not in stalled.num_scheduled_tokens
    assert stalled.layered_prefill_plan is None
    assert stalled.num_scheduled_tokens[first.request_id] == 1
    assert stalled.num_scheduled_tokens[second.request_id] == 1
    assert first.num_computed_tokens == before[first.request_id] + 1
    assert second.num_computed_tokens == before[second.request_id] + 1
    assert scheduler._layered_fallback_counts.get("kv_alloc_failed") == 1
    assert third in scheduler.waiting or third in scheduler.skipped_waiting
    assert third.status.name == "WAITING"

    from vllm.v1.outputs import ModelRunnerOutput

    admitted = None
    for _ in range(6):
        step = stalled if stalled is not None else scheduler.schedule()
        stalled = None
        if (
            third.request_id in step.num_scheduled_tokens
            and step.layered_prefill_plan is not None
            and step.layered_prefill_plan.group_id == 0
        ):
            admitted = step
            break
        live_ids = [
            req.request_id
            for req in (first, second)
            if not req.is_finished() and req.request_id in step.num_scheduled_tokens
        ]
        assert live_ids, step.num_scheduled_tokens
        assert third.request_id not in step.num_scheduled_tokens
        req_ids = list(step.num_scheduled_tokens)
        scheduler.update_from_output(
            step,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
                sampled_token_ids=[[1100] for _ in req_ids],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
    assert admitted is not None
    assert third.status.name == "RUNNING"
    # req 0 finished and freed a block. req 1 is a new rider of req 2,
    # not a slot left behind by the failed admission.
    assert first.is_finished()
    assert not first.layered_fused_decode_slot
    assert second.layered_fused_decode_slot
    assert second.layered_fused_decode_owner == third.request_id
    assert second.layered_fused_resume_group == 1


def test_orphaned_fused_rider_resumes_next_group_not_layer0():
    """A rider that already ran group 0 finishes later groups in place."""
    from tests.v1.core.utils import create_requests

    scheduler = _local_scheduler(max_num_batched_tokens=32)
    _enable_layered_scheduler(
        scheduler, fuse_mixed_batch=True, allowed_num_groups=[4]
    )
    first, second = create_requests(num_requests=2, num_tokens=8, max_tokens=4)
    scheduler.add_request(first)
    _drive_layered_until_sample(scheduler, lambda _req_id: 1000)

    scheduler.add_request(second)
    mixed = scheduler.schedule()
    assert mixed.layered_prefill_plan is not None
    assert mixed.layered_prefill_plan.group_id == 0
    assert first.request_id in mixed.num_scheduled_tokens
    cursor = first.num_computed_tokens
    scheduler.update_from_output(
        mixed, _empty_runner_output(list(mixed.num_scheduled_tokens))
    )
    assert first.layered_fused_decode_slot
    assert first.layered_fused_resume_group == 1
    assert first.num_computed_tokens == cursor

    # Owner cohort is gone; the rider must not be replayed from layer 0.
    second.layered_prefill_enabled = False
    second.layered_fused_decode_ids = None

    resumed = scheduler.schedule()
    assert resumed.layered_prefill_plan is not None
    assert resumed.layered_prefill_plan.group_id == 1
    assert resumed.layered_prefill_plan.prefill_req_ids == (first.request_id,)
    assert not resumed.layered_prefill_plan.is_sampling_step
    assert resumed.num_scheduled_tokens == {first.request_id: 1}
    assert first.num_computed_tokens == cursor
    assert first.layered_fused_decode_slot
    assert first.layered_fused_resume_group == 2
    assert second.request_id not in resumed.num_scheduled_tokens
