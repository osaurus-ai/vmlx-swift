#!/usr/bin/env python3
"""MiniMax-M3 aligned-chunk prefix cache: the vMLX Python runtime's per-turn cached / prefilled token counts, for the
Swift port's parity test (Tests/MLXLMTests/MiniMaxM3AlignedPrefixCacheTests.swift).

Run against a CLEAN export of the Python branch commit (never a working tree):

    git -C <vmlx> archive ef84c03f | tar -x -C /tmp/py_ef84
    MLX_ENABLE_TF32=0 PYTHONPATH=/tmp/py_ef84 python generate_aligned_prefix_fixtures.py \\
        --python-root /tmp/py_ef84 --commit ef84c03f --out <fixture.json> [--weights <dir>]

It drives the Python test's own machinery (tests/test_minimax_m3_aligned_prefix_cache.py: the tiny synthetic M3 with the
released MSA geometry, the real Scheduler on the block-disk tier, the cold arm = _bypass_prefix_cache) through the same
scenarios the Swift test runs, and records the prompts (token ids), the per-turn cached / prefilled counts and each
turn's greedy tokens and a digest of the first-step logits. Counts depend only on the prompt lengths and the rule, so
the Swift test compares them on its own model; `--weights` also writes the model's unquantized parameters (not part of
the repo) for an opt-in same-weights comparison of tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-root", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()
    sys.path.insert(0, args.python_root)
    os.chdir(args.python_root)

    import mlx.core as mx
    import numpy as np
    from _pytest.monkeypatch import MonkeyPatch
    from mlx.utils import tree_flatten

    import tests.test_minimax_m3_aligned_prefix_cache as A
    import tests.test_minimax_m3_dspark as T
    from vmlx_engine.scheduler import Scheduler

    mp = MonkeyPatch()
    for name in ("VMLX_M3_ALIGNED_PREFIX", "VMLX_DSPARK", "VMLX_DSPARK_MAX_DRAFT", "VMLX_DSPARK_WIDTH",
                 "VMLX_DSPARK_SAMPLED", "VMLX_DSPARK_AR_FALLBACK", "VMLX_DISABLE_GEN_PROMPT_STRIP"):
        mp.delenv(name, raising=False)
    mp.setattr(Scheduler, "_should_clean_prefill_over_disk_only",
               lambda self, **kw: (False, {"cost_reason": "pinned_by_test"}))

    original = T._config
    T._config = lambda block=128, topk=16: A._m3_config()
    try:
        model = T._model(seed=5, block=128, topk=16)
        if args.weights:
            raw = T._model(seed=5, block=128, topk=16, quantize=False)
            Path(args.weights).mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(str(Path(args.weights) / "model.safetensors"), dict(tree_flatten(raw.parameters())))
            (Path(args.weights) / "config.json").write_text(json.dumps(A._m3_config(), indent=1))
    finally:
        T._config = original

    trace = A._Trace(mp)

    def digest(x) -> str:
        a = np.array(mx.array(x).astype(mx.float32))
        return hashlib.sha256(a.tobytes()).hexdigest()

    def row(res, prompt, prev=None):
        return {"prompt": len(prompt), "cached": res["cached"], "prefill": len(prompt) - res["cached"],
                "boundary": A._boundary(prompt), "prev_boundary": None if prev is None else A._boundary(prev),
                "tokens": res["tokens"], "first_logits_sha256": digest(res["logits"][0])}

    out = {"python_commit": args.commit, "vocab": A.VOCAB, "gen": A.GEN, "end": A.END, "max_tokens": A.MAX_TOKENS,
           "tool_lengths": list(A.TOOL_LENS)}
    first, tools = A._loop_prompts()
    out["first_prompt"] = first
    out["tools"] = tools

    # (a)+(b) the 6-turn tool loop, exact repeat, (c) the partial hit -- block-disk tier, steps 320 and 2048.
    out["loops"] = {}
    for step in (320, 2048):
        with tempfile.TemporaryDirectory() as tmp:
            sched = A._scheduler(model, Path(tmp), step)
            try:
                prompt, prev, turns = list(first), None, []
                for turn in range(6):
                    warm = A._run(sched, trace, prompt)
                    cold = A._run(sched, trace, prompt, bypass=True)
                    assert A._same(warm, cold), (step, turn)
                    r = row(warm, prompt, prev)
                    r["cold_tokens"] = cold["tokens"]
                    turns.append(r)
                    prev = prompt
                    if turn < 5:
                        prompt = prompt + warm["tokens"] + A.END + tools[turn] + A.GEN
                rep = A._run(sched, trace, prev)
                cut = len(prev) - len(A.GEN) - A.TOOL_LENS[4] + 213
                tail = A._ids(len(prev) - len(A.GEN) - cut, 777)
                variant = prev[:cut] + tail + A.GEN
                part = A._run(sched, trace, variant)
                assert A._same(part, A._run(sched, trace, variant, bypass=True))
                out["loops"][str(step)] = {
                    "turns": turns, "exact_repeat": row(rep, prev),
                    "partial": {"divergence": cut, "tail": tail, **row(part, variant)}}
            finally:
                sched.shutdown()

    # (d) short prompt, extension, a long prompt's continuation, then (e) a restart onto the same SSD store.
    with tempfile.TemporaryDirectory() as tmp:
        step = 320
        sched = A._scheduler(model, Path(tmp), step)
        try:
            short = A._ids(211, 7) + A.GEN
            s_cold = A._run(sched, trace, short)
            s_again = A._run(sched, trace, short)
            ext_tail = A._ids(97, 8)
            longer = short + s_cold["tokens"] + A.END + ext_tail + A.GEN
            ext = A._run(sched, trace, longer)
            long_prompt = A._ids(2390, 9) + A.GEN
            first_long = A._run(sched, trace, long_prompt)
            nxt_tail = A._ids(611, 10)
            nxt = long_prompt + first_long["tokens"] + A.END + nxt_tail + A.GEN
            warm_ram = A._run(sched, trace, nxt)
        finally:
            sched.shutdown()
        sched = A._scheduler(model, Path(tmp), step)
        try:
            from_disk = A._run(sched, trace, nxt)
            cold = A._run(sched, trace, nxt, bypass=True)
            third_tail = A._ids(150, 11)
            third = nxt + cold["tokens"] + A.END + third_tail + A.GEN
            cont = A._run(sched, trace, third)
        finally:
            sched.shutdown()
        out["short_and_restart"] = {
            "step": step, "short": short, "extension_tail": ext_tail, "long": long_prompt, "next_tail": nxt_tail,
            "third_tail": third_tail,
            "rows": {"short_cold": row(s_cold, short), "short_repeat": row(s_again, short),
                     "extension": row(ext, longer), "long_first": row(first_long, long_prompt),
                     "next_warm_ram": row(warm_ram, nxt, long_prompt), "restart_exact_from_disk": row(from_disk, nxt),
                     "restart_continuation": row(cont, third, nxt)}}

    # The whole-prompt memory tier (Python's tier shaped like Swift's whole-entry disk tier), steps 320 and 1000.
    out["memory_tier"] = {}
    from vmlx_engine.scheduler import SchedulerConfig

    # Six turns, the exact repeat and the same partial hit as the block-disk loops (whole entries have no blocks to
    # match inside a longer entry, so the partial hit reuses the last chunk boundary of an entry that IS a prefix).
    for step in (320, 1000):
        sched = Scheduler(model, A._Tok(), SchedulerConfig(
            max_num_seqs=1, prefill_step_size=step, enable_prefix_cache=True, use_paged_cache=False,
            enable_block_disk_cache=False, use_memory_aware_cache=True))
        try:
            prompt, prev, turns = list(first), None, []
            for turn in range(6):
                warm = A._run(sched, trace, prompt)
                cold = A._run(sched, trace, prompt, bypass=True)
                assert A._same(warm, cold), (step, turn)
                turns.append(row(warm, prompt, prev))
                prev = prompt
                if turn < 5:
                    prompt = prompt + warm["tokens"] + A.END + tools[turn] + A.GEN
            rep = A._run(sched, trace, prev)
            cut = len(prev) - len(A.GEN) - A.TOOL_LENS[4] + 213
            tail = A._ids(len(prev) - len(A.GEN) - cut, 777)
            variant = prev[:cut] + tail + A.GEN
            part = A._run(sched, trace, variant)
            assert A._same(part, A._run(sched, trace, variant, bypass=True))
            out["memory_tier"][str(step)] = {
                "turns": turns, "exact_repeat": row(rep, prev),
                "partial": {"divergence": cut, "tail": tail, **row(part, variant)}}
        finally:
            sched.shutdown()

    mp.undo()
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    for step, loop in out["loops"].items():
        print(step, [(t["prompt"], t["cached"], t["prefill"]) for t in loop["turns"]],
              "repeat", loop["exact_repeat"]["cached"], "partial", loop["partial"]["divergence"],
              loop["partial"]["cached"])
    print("short/restart", {k: (v["prompt"], v["cached"]) for k, v in out["short_and_restart"]["rows"].items()})
    for step, loop in out["memory_tier"].items():
        print("memory", step, [(t["prompt"], t["cached"]) for t in loop["turns"]], "repeat",
              loop["exact_repeat"]["cached"], "partial", loop["partial"]["divergence"], loop["partial"]["cached"])


if __name__ == "__main__":
    main()
