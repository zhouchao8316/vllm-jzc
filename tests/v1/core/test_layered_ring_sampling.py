# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract test for the scheduler's global sampling decision.

Run directly to avoid importing the GPU runtime. Execute the real scheduler
method, replacing only the RingStepPlan builder with an argument recorder.
"""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch


class TestLayeredRingSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        source = root / "vllm/v1/core/sched/scheduler.py"
        tree = ast.parse(source.read_text())
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_attach_ring_step_plan"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                method,
            ],
            type_ignores=[],
        )
        env: dict = {}
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), env)
        cls.attach = staticmethod(env["_attach_ring_step_plan"])

    def test_sampling_modes(self):
        # sample_p, with_d, d_finishing, fused, expected sampling_step.
        cases = [
            (False, True, False, False, True),  # Serial intermediate + D.
            (False, True, True, False, False),  # Finishing D needs no recv.
            (False, False, False, False, False),  # P-only intermediate.
            (False, True, False, True, False),  # Fuse riders still held.
            (True, True, False, True, True),  # Fuse final samples.
            (True, False, False, False, True),  # P-only final.
            (None, True, False, False, None),  # Regular chunk unchanged.
        ]
        fake = NS(build_ring_step_plan=lambda **kw: NS(**kw))
        for sample_p, with_d, finishing, fused, expected in cases:
            with self.subTest(case=(sample_p, with_d, finishing, fused)):
                rows = {"p": 1024}
                requests = {"p": NS(num_prompt_tokens=1024, max_tokens=32)}
                cursors = [0]
                if with_d:
                    rows["d"] = 1
                    requests["d"] = NS(
                        num_prompt_tokens=16,
                        max_tokens=1 if finishing else 32,
                    )
                    cursors.append(16)
                output = NS(
                    num_scheduled_tokens=rows,
                    scheduled_new_reqs=[],
                    scheduled_cached_reqs=NS(
                        req_ids=list(rows),
                        num_computed_tokens=cursors,
                    ),
                    layered_prefill_plan=None
                    if sample_p is None
                    else NS(is_sampling_step=sample_p, prefill_req_ids=("p",)),
                )
                scheduler = NS(
                    parallel_config=NS(pipeline_parallel_size=4),
                    use_v2_model_runner=True,
                    requests=requests,
                    num_spec_tokens=0,
                    _ring_step_seq=0,
                    _fuse_mixed_batch_active=lambda: fused,
                )
                with patch.dict(sys.modules, {"vllm.v1.worker.gpu.pp_utils": fake}):
                    self.attach(scheduler, output)
                self.assertEqual(output.ring_step_plan.sampling_step, expected)
                self.assertEqual(
                    tuple(output.ring_step_plan.request_order), tuple(rows)
                )


if __name__ == "__main__":
    unittest.main()
