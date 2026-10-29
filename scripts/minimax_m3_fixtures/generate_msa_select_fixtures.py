#!/usr/bin/env python3
"""Golden digests for the Swift MiniMax-M3.1 MSA selection kernels (`MiniMaxM3MSA.blockScores` / `kernelBlockScores`).

The vMLX Python runtime's selection (`vmlx_engine/models/minimax_m3/msa.py`, perf/minimax-m3-msa-select 5fc31aca)
computes the indexer's block scores [B, G, Sq, n_blocks] with the score-sheet ops (`reference_block_scores`) or with a
Metal kernel that reproduces them bit for bit (`kernel_block_scores`, `nax` with TF32 on, `fma` with TF32 off). This
script runs both on synthetic index tensors at the shapes of the Python tests (`tests/test_minimax_m3_msa_select.py`)
and records SHA-256 digests (raw bytes, row-major) of:
  - the inputs (index queries / keys);
  - the reference block scores, the routed (kernel) block scores and the top-k block ids (argpartition order);
  - the path the router took (`nax` / `fma`) and the admitted arithmetic;
plus the SHA-256 of every Metal source of msa.py (one leading / trailing newline dropped), which the Swift test checks
against its own sources. Nothing here reads model weights.

Inputs are U[-2, 2) from MLX's counter-based PRNG with explicit keys, on the CPU (integer bits and correctly rounded
IEEE ops only), so the Swift test regenerates exactly the same tensors (it checks their digests first). The index keys
drift slowly (a cumsum term) so neighbouring blocks score alike and the top-16 boundary is contested.

Cases (key counts are not multiples of 128, 256 or 2048):
  rows     1 / 2 / 5 / 16 rows ending at Sk 2049 / 2853 / 3011 / 8268 / 32845, G 4 / 8 (decode, verify-sized);
  chunk    2048 / 1000 / 512-row chunks at Sk 2853 / 3011 / 8268 / 32845, G 4 / 8 (rows before key 2048 have fewer
           than 16 causal blocks: the -inf future blocks tie);
  batch    batch 2 with 6 rows; strided queries and keys read through a view of a longer buffer with a NaN tail;
  ties     index keys repeating every block (every fully causal block of a row scores exactly the same);
  nan      a NaN index key inside the rows' causal range and one past some rows' positions;
  dtypes   float16 / float32 decode, verify and chunk cases.
Both TF32 modes (MLX runs float32 matmuls as TF32 on M5 unless MLX_ENABLE_TF32=0 is set before it initializes): the
generator runs once per mode in a subprocess and records both (`strict`, `tf32`).

    <python with mlx 0.32.3> scripts/minimax_m3_fixtures/generate_msa_select_fixtures.py --vmlx <vmlx export> \\
        --vmlx-commit <sha> --output Tests/MLXLMTests/Fixtures/MiniMaxM3/msa-select.json

Fails closed: every case's kernel scores equal the reference bit for bit, the reference is a real selection (local
block +inf in every row, finite scores, the set moves when block ids shift by one), the router took the admitted kernel.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

D, BLOCK, TOPK = 128, 128, 16
ROWS = [(sk, g, r) for g in (4, 8) for sk in (2049, 2853, 3011, 8268, 32845) for r in (1, 2, 5, 16)]
CHUNK = [(sk, g, r) for g in (4, 8) for sk, r in ((2853, 2048), (3011, 1000), (3011, 512), (8268, 512),
                                                  (8268, 2048), (32845, 2048))]
DTYPES = [("float16", sk, 4, r) for sk, r in ((2853, 1), (3011, 5), (3011, 512))] + \
         [("float32", sk, 4, r) for sk, r in ((2853, 1), (3011, 5), (3011, 512))]


def fail(message):
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def kernel_sources(msa_py: Path) -> dict:
    out = {}
    for node in ast.parse(msa_py.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            n = node.targets[0].id
            if n.endswith("_SOURCE") or n == "_NAX_HEADER":
                v = ast.literal_eval(node.value)
                v = v[1:] if v.startswith("\n") else v
                v = v[:-1] if v.endswith("\n") else v
                out[n] = hashlib.sha256(v.encode()).hexdigest()
    return out


def worker(mode: str, out_path: Path):
    import numpy as np
    import mlx.core as mx

    from vmlx_engine.models.minimax_m3 import msa

    def digest(x) -> str:
        x = mx.contiguous(x)
        mx.eval(x)
        a = np.array(x.view(mx.uint16)) if x.dtype == mx.bfloat16 else np.array(x)
        return hashlib.sha256(a.tobytes()).hexdigest()

    def uniform(n: int, shape):
        return mx.random.uniform(-2.0, 2.0, shape, key=mx.random.key(n), stream=mx.cpu)

    def index(seed: int, sk: int, groups: int, rows: int, dtype, batch: int = 1):
        base = seed * 100
        with mx.stream(mx.cpu):
            ik = (uniform(base + 1, (batch, 1, sk, D))
                  + mx.cumsum(uniform(base + 2, (batch, 1, sk, D)) * 0.05, axis=2)).astype(dtype)
            iq = uniform(base + 3, (batch, groups, rows, D)).astype(dtype)
            mx.eval(ik, iq)
        return iq, ik

    def top(scores):
        return mx.argpartition(-scores, kth=TOPK - 1, axis=-1)[..., :TOPK]

    picks = []
    real_choose = msa.choose_select_path

    def spy(*a, **kw):
        p = real_choose(*a, **kw)
        picks.append(p)
        return p

    msa.choose_select_path = spy
    arith = msa.select_arithmetic(mx.bfloat16, D)
    if arith not in ("nax", "fma"):
        fail(f"{mode}: no selection kernel admitted")

    def check(label, ref, got, rt, offset):
        r, g = np.array(ref), np.array(got)
        rt_ = np.array(rt).astype(np.int64)
        B, G, L, nb = r.shape
        if nb <= TOPK:
            fail(f"{label}: no selection")
        local = (offset + np.arange(L)) // BLOCK
        if not np.isposinf(r[:, :, np.arange(L), local]).all() or not np.isfinite(r).any():
            fail(f"{label}: degenerate reference")
        if np.array_equal(np.sort(rt_, -1), np.sort((rt_ + 1) % nb, -1)):
            fail(f"{label}: degenerate selection")
        if not np.array_equal(r, g, equal_nan=True):
            fail(f"{label}: kernel scores differ from the reference")

    def case(label, iq, ik, offset, extra=None, q_run=None, k_run=None):
        """Reference + routed scores of (iq, ik) (the routed call may read other views of the same values)."""
        ref = msa.reference_block_scores(iq, ik, offset, BLOCK, 1)
        rt = top(ref)
        mx.eval(ref, rt)
        picks.clear()
        got = msa.block_scores(q_run if q_run is not None else iq, k_run if k_run is not None else ik, offset,
                               BLOCK, 1)
        gt = top(got)
        mx.eval(got, gt)
        if picks != [arith]:
            fail(f"{label}: router took {picks}, admitted {arith}")
        check(label, ref, got, rt, offset)
        if not np.array_equal(np.array(rt), np.array(gt)):
            fail(f"{label}: top-k differs")
        e = {"label": label, "offset": offset, "path": picks[0], "iq": digest(iq), "ik": digest(ik),
             "reference": digest(ref), "kernel": digest(got), "top": digest(rt), "blocks": int(ref.shape[-1])}
        if extra:
            e.update(extra)
        return e

    report = {"mode": mode, "arithmetic": arith,
              "arithmetic_by_dtype": {dt: msa.select_arithmetic(getattr(mx, dt), D)
                                      for dt in ("bfloat16", "float16", "float32")},
              "matrix_units": bool(msa.nax_available()), "rows": [], "chunk": [], "dtypes": [], "special": []}
    for sk, g, r in ROWS:
        seed = 1000 + sk * 10 + g + r * 100_000
        iq, ik = index(seed, sk, g, r, mx.bfloat16)
        report["rows"].append(case(f"rows {r} Sk{sk} G{g}", iq, ik, sk - r, {"sk": sk, "groups": g, "rows": r,
                                                                            "seed": seed}))
    print(f"[{mode}] rows: {len(report['rows'])}", flush=True)
    for sk, g, r in CHUNK:
        seed = 2000 + sk * 10 + g + r * 100_000
        iq, ik = index(seed, sk, g, r, mx.bfloat16)
        report["chunk"].append(case(f"chunk {r} Sk{sk} G{g}", iq, ik, sk - r, {"sk": sk, "groups": g, "rows": r,
                                                                              "seed": seed}))
        mx.clear_cache()
    print(f"[{mode}] chunk: {len(report['chunk'])}", flush=True)
    for dt, sk, g, r in DTYPES:
        seed = 3000 + sk * 10 + g + r * 100_000
        iq, ik = index(seed, sk, g, r, getattr(mx, dt))
        report["dtypes"].append(case(f"{dt} {r} Sk{sk} G{g}", iq, ik, sk - r,
                                     {"dtype": dt, "sk": sk, "groups": g, "rows": r, "seed": seed}))
    # batch 2, 6 rows
    iq, ik = index(4001, 3011, 4, 6, mx.bfloat16, batch=2)
    report["special"].append(case("batch2", iq, ik, 3011 - 6, {"kind": "batch2", "seed": 4001}))
    # strided queries and a key view over a NaN tail; then each row alone over its prefix (DSpark verify)
    sk, rows = 8268, 9
    iq, ik = index(4002, sk, 4, rows, mx.bfloat16)
    iq_t = mx.contiguous(iq.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    buf = mx.concatenate([ik, mx.full((1, 1, 300, D), mx.nan, dtype=ik.dtype)], axis=2)
    view = buf[:, :, :sk]
    e = case("strided", iq, ik, sk - rows, {"kind": "strided", "seed": 4002}, q_run=iq_t, k_run=view)
    e["one_row"] = [case(f"strided row {r}", iq[:, :, r:r + 1], ik[:, :, :sk - rows + r + 1], sk - rows + r,
                         q_run=iq_t[:, :, r:r + 1], k_run=view[:, :, :sk - rows + r + 1])["top"] for r in range(rows)]
    report["special"].append(e)
    # exact ties at the top-16 boundary: one block of keys repeated
    for sk, r in ((2853, 1), (3011, 5), (3011, 600)):
        seed = 4100 + r
        with mx.stream(mx.cpu):
            one = uniform(seed * 100 + 1, (BLOCK, D))
            ik = mx.tile(one, (-(-sk // BLOCK), 1))[:sk][None, None].astype(mx.bfloat16)
            iq = uniform(seed * 100 + 3, (1, 4, r, D)).astype(mx.bfloat16)
            mx.eval(ik, iq)
        s = np.array(msa.reference_block_scores(iq, ik, sk - r, BLOCK, 1))
        srt = -np.sort(-s, axis=-1)
        full = (sk - r + np.arange(r)) >= (TOPK + 1) * BLOCK
        if not full.any() or not (srt[:, :, full, TOPK - 1] == srt[:, :, full, TOPK]).all():
            fail(f"ties Sk{sk} r{r}: no tie at the top-k boundary")
        report["special"].append(case(f"ties {r} Sk{sk}", iq, ik, sk - r, {"kind": "ties", "sk": sk, "rows": r,
                                                                          "seed": seed}))
    # NaN propagation
    sk, rows = 3011, 5
    iq, ik = index(4200, sk, 4, rows, mx.bfloat16)
    pos = mx.arange(sk).reshape(1, 1, sk, 1)
    dim = mx.arange(D).reshape(1, 1, 1, D)
    poison = ((pos == 700) & (dim == 3)) | ((pos == sk - 2) & (dim == 0))
    ikn = mx.where(poison, mx.array(float("nan"), dtype=ik.dtype), ik)
    e = case("nan", iq, ikn, sk - rows, {"kind": "nan", "seed": 4200})
    report["special"].append(e)
    out_path.write_text(json.dumps(report))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmlx", required=True, type=Path)
    ap.add_argument("--vmlx-commit", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--worker", choices=["strict", "tf32"])
    ap.add_argument("--worker-out", type=Path)
    a = ap.parse_args()
    sys.path.insert(0, str(a.vmlx))
    if a.worker:
        worker(a.worker, a.worker_out)
        return

    import mlx.core as mx

    modes = {}
    for mode in ("strict", "tf32"):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(a.vmlx) + os.pathsep + env.get("PYTHONPATH", "")
        if mode == "strict":
            env["MLX_ENABLE_TF32"] = "0"
        else:
            env.pop("MLX_ENABLE_TF32", None)
        part = a.output.with_suffix(f".{mode}.part.json")
        subprocess.run([sys.executable, __file__, "--vmlx", str(a.vmlx), "--vmlx-commit", a.vmlx_commit,
                        "--output", str(a.output), "--worker", mode, "--worker-out", str(part)], env=env, check=True)
        modes[mode] = json.loads(part.read_text())
        part.unlink()
    for kind in ("rows", "chunk", "dtypes", "special"):
        for s, t in zip(modes["strict"][kind], modes["tf32"][kind]):
            if (s["iq"], s["ik"]) != (t["iq"], t["ik"]):
                fail(f"{kind} {s['label']}: inputs differ between the TF32 modes")
    src = a.vmlx / "vmlx_engine/models/minimax_m3/msa.py"
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    fixture = {
        "generator": "scripts/minimax_m3_fixtures/generate_msa_select_fixtures.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "vmlx_commit": a.vmlx_commit,
        "msa_py_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "kernel_sources_sha256": kernel_sources(src),
        "mlx": mx.__version__,
        "python": platform.python_version(),
        "gpu": str(info.get("architecture", "")),
        "geometry": {"index_head_dim": D, "block": BLOCK, "topk": TOPK, "local": 1},
        "modes": modes,
    }
    a.output.write_text(json.dumps(fixture, indent=None, separators=(",", ":")))
    print(f"wrote {a.output} ({a.output.stat().st_size} bytes); arithmetic strict {modes['strict']['arithmetic']}, "
          f"tf32 {modes['tf32']['arithmetic']}")


if __name__ == "__main__":
    main()
