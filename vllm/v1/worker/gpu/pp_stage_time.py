# SPDX-License-Identifier: Apache-2.0
"""Opt-in PP stage timing.

Set ``VLLM_PP_STAGE_TIME`` to a directory. Each rank appends JSON lines.
``compute`` is the stage forward. ``recv_wait`` and ``send_wait`` are the
pipeline stalls around it. Nothing here is read back on the hot path, and
device spans do not synchronize.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from contextlib import contextmanager

_DIR = os.environ.get("VLLM_PP_STAGE_TIME", "").strip()
_ENABLED = bool(_DIR)
_RECORDER: "Recorder | None" = None


def _shape(scheduler_output) -> tuple[int, int, int, int] | None:
    tokens = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
    counts = [int(v) for v in tokens.values()]
    if not counts:
        return None
    prefill = sum(v for v in counts if v > 1)
    decode = sum(1 for v in counts if v == 1)
    return len(counts), sum(counts), prefill, decode


class Recorder:
    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.disabled = False
        self._pp = -1
        self._tp = -1
        self._ranked = False
        self._buf: list[tuple] = []
        self._rows: list[dict] = []
        os.makedirs(directory, exist_ok=True)
        atexit.register(self.flush)

    def _ranks(self) -> bool:
        if self._ranked or self.disabled:
            return not self.disabled
        try:
            from vllm.distributed.parallel_state import get_pp_group, get_tp_group

            self._pp = int(get_pp_group().rank_in_group)
            self._tp = int(get_tp_group().rank_in_group)
            self._ranked = True
        except Exception:
            self.disabled = True
        return not self.disabled

    def host(self, seg: str, scheduler_output, ms: float, wall: float) -> None:
        shape = _shape(scheduler_output)
        if shape is None or not self._ranks():
            return
        width, total, prefill, decode = shape
        self._rows.append(
            {
                "pp": self._pp,
                "tp": self._tp,
                "seg": seg,
                "clock": "host",
                "ms": float(ms),
                "t": wall,
                "width": width,
                "tokens": total,
                "p_tok": prefill,
                "d_req": decode,
            }
        )
        if len(self._rows) >= 32:
            self._write()

    def begin(self, seg: str, scheduler_output):
        self._drain()
        shape = _shape(scheduler_output)
        if shape is None or not self._ranks():
            return None
        try:
            import torch

            stream = torch.npu.current_stream()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record(stream)
        except Exception:
            self.disabled = True
            return None
        width, total, prefill, decode = shape
        return (start, end, seg, time.time(), width, total, prefill, decode)

    def end(self, token) -> None:
        if token is None or self.disabled:
            return
        start, end, seg, wall, width, total, prefill, decode = token
        try:
            import torch

            end.record(torch.npu.current_stream())
        except Exception:
            self.disabled = True
            return
        self._buf.append((start, end, seg, wall, width, total, prefill, decode))

    def _drain(self) -> None:
        if not self._buf or self.disabled:
            return
        ready = []
        pending = []
        for item in self._buf:
            try:
                done = bool(item[1].query())
            except Exception:
                self.disabled = True
                self._buf = []
                return
            if done:
                ready.append(item)
            else:
                pending.append(item)
        self._buf = pending
        for start, end, seg, wall, width, total, prefill, decode in ready:
            try:
                ms = float(start.elapsed_time(end))
            except Exception:
                self.disabled = True
                return
            self._rows.append(
                {
                    "pp": self._pp,
                    "tp": self._tp,
                    "seg": seg,
                    "clock": "npu",
                    "ms": ms,
                    "t": wall,
                    "width": width,
                    "tokens": total,
                    "p_tok": prefill,
                    "d_req": decode,
                }
            )
        if len(self._rows) >= 32:
            self._write()

    def _write(self) -> None:
        if not self._rows or self._pp < 0:
            return
        path = os.path.join(self.directory, f"pp{self._pp}_tp{self._tp}.jsonl")
        payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in self._rows)
        self._rows = []
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(payload)

    def flush(self) -> None:
        if self.disabled:
            return
        if self._buf:
            try:
                import torch

                torch.npu.current_stream().synchronize()
            except Exception:
                return
            self._drain()
        self._write()


def _recorder() -> Recorder:
    global _RECORDER
    if _RECORDER is None:
        _RECORDER = Recorder(_DIR)
    return _RECORDER


@contextmanager
def host_span(seg: str, scheduler_output, dummy_run: bool = False):
    if not _ENABLED or dummy_run or scheduler_output is None:
        yield
        return
    wall = time.time()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _recorder().host(seg, scheduler_output, (time.perf_counter() - t0) * 1000.0, wall)


@contextmanager
def device_span(seg: str, scheduler_output, dummy_run: bool = False):
    if not _ENABLED or dummy_run or scheduler_output is None:
        yield
        return
    rec = _recorder()
    token = rec.begin(seg, scheduler_output)
    try:
        yield
    finally:
        rec.end(token)
