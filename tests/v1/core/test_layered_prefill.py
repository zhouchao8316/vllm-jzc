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
    assert first.reuse_kv_blocks is False
    request.layered_prefill_group_id += 1
    request.layered_prefill_kv_reserved = True
    second = policy.make_plan(request)
    assert second.group_id == 1
    assert second.commit_tokens["req"] == 513
    assert second.reuse_kv_blocks is True


def test_policy_aligns_pp_groups_with_stage_partitions():
    config = _config(layers=8)
    config.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    policy = LayeredPrefillPolicy(config)
    request = SimpleNamespace(
        request_id="req",
        num_prompt_tokens=1,
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


def test_config_rejects_multiple_groups_per_step_in_phase_one():
    with pytest.raises(ValueError, match="max_groups_per_step=1"):
        LayeredPrefillConfig(enabled=True, max_groups_per_step=2)
