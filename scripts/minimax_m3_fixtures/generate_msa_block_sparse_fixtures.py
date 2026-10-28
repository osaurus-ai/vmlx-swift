#!/usr/bin/env python3
"""Golden digests for the Swift MiniMax-M3.1 block-sparse MSA port (`Libraries/MLXLLM/Models/MiniMaxM3MSA.swift`).

The vMLX Python runtime (`vmlx_engine/models/minimax_m3/msa.py`, perf/minimax-m3-block-sparse-msa) runs every MSA path on
synthetic tensors at the released attention geometry (64 query heads, 4 or 8 KV groups, head dim 128, index heads of
128, blocks of 128, top-16, local 1). Nothing here reads model weights.

Inputs come from MLX's counter-based PRNG with explicit keys (uniform on the CPU: integer bits and correctly rounded
IEEE ops only), so the Swift test regenerates exactly the same tensors (it checks their SHA-256 first). For each case the fixture records SHA-256 digests (raw bytes, row-major) of:
  - the inputs;
  - the selection (`Indexer.select` block ids, argpartition order: the decode kernel merges chunks in that order);
  - every path's output: `dense` (the mask path), `decode` (split-key decode kernel), `gather` (MLX ops), `kernel`
    (chunk kernel: matrix units for bf16, simdgroup matrices for fp32) and `kernel_sg` (the simdgroup kernel forced
    on bf16), plus the path the router picks on its own (`auto`);
and each output's relative error against a float64 NumPy reference (key j allowed for row r when its block is selected
and j <= offset + r), per row over the H * D output: median / max.

Cases (key counts are not multiples of 128 or 2048):
  decode   one row at position p over keys 0..p: Sk 2049 / 2853 / 3011 / 8268, G 4 / 8, the last three rows plus 6
           random positions >= 2048;
  verify   2-16 rows ending at Sk in one call, one selection for all rows: widths 2 / 5 / 8 / 9 / 16;
  chunk    101 / 512 rows ending at Sk (causal inside the chunk, future blocks in the selection).
Both float32 and bfloat16. MLX on M5 runs float32 matmuls (the indexer scores, dense SDPA) as TF32 unless
MLX_ENABLE_TF32=0 is set before MLX initializes, so the generator runs once per mode in a subprocess and records both
(`strict`, `tf32`); the Swift test reads the mode its own process runs in.

    PYTHONPATH=<vmlx checkout> <python with mlx 0.32.3> \\
        scripts/minimax_m3_fixtures/generate_msa_block_sparse_fixtures.py --vmlx <vmlx checkout> \\
        --vmlx-commit <sha> --output Tests/MLXLMTests/Fixtures/MiniMaxM3/msa-block-sparse.json

Fails closed: every float64 reference finite, the selection sparse, every path within the Python tests' bounds
(fp32 2e-5, bf16 6e-3 per row vs float64; fp32 under TF32 6e-3), the groups' selections not all equal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

H, D, IDX_D, BLOCK, TOPK = 64, 128, 128, 128, 16
FP32_TOL, BF16_TOL = 2e-5, 6e-3
TF32_TOL = 6e-3   # float32 under TF32 matmuls (dense SDPA over several rows): ~1e-3 measured

DECODE = [(sk, g) for g in (4, 8) for sk in (2049, 2853, 3011, 8268)]
VERIFY = [(sk, g, w) for g in (4, 8) for sk in (2853, 8268) for w in (2, 5, 8, 9, 16)]
CHUNK = [(2049, 4, 101), (2853, 4, 512), (3011, 4, 101), (8268, 4, 512), (8268, 4, 101), (2853, 8, 512),
         (8268, 8, 101)]
DTYPES = ("float32", "bfloat16")


def fail(message):
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def seed_of(kind: str, sk: int, groups: int, extra: int = 0) -> int:
    """The PRNG seed of a case; shared by both dtypes (the bf16 inputs are the fp32 ones rounded)."""
    return {"decode": 1, "verify": 2, "chunk": 3}[kind] * 100_000 + groups * 10_000 + sk + extra * 1_000_000


def worker(mode: str, out_path: Path):
    import numpy as np
    import mlx.core as mx

    from vmlx_engine.models.minimax_m3 import msa
    from vmlx_engine.models.minimax_m3.minimax_m3 import Attention, ModelArgs

    def key(n: int):
        return mx.random.key(n)

    def digest(x) -> str:
        x = mx.contiguous(x)
        mx.eval(x)
        a = np.array(x.view(mx.uint16)) if x.dtype == mx.bfloat16 else np.array(x)
        return hashlib.sha256(a.tobytes()).hexdigest()

    def uniform(n: int, shape):
        """U[-2, 2) from PRNG key n, on the CPU: bits -> IEEE divide / multiply / add only. (MLX's normal sampler runs
        erfinv, whose last bits differ between MLX builds; the GPU's divide is not correctly rounded.)"""
        return mx.random.uniform(-2.0, 2.0, shape, key=key(n), stream=mx.cpu)

    def context(seed: int, sk: int, groups: int, dtype):
        """K / V [1, G, sk, D] and index keys [1, 1, sk, IDX_D] (a slow drift so neighbouring blocks score alike)."""
        base = seed * 100_000
        with mx.stream(mx.cpu):
            k = uniform(base + 1, (1, groups, sk, D)).astype(dtype)
            v = uniform(base + 2, (1, groups, sk, D)).astype(dtype)
            ik = (uniform(base + 3, (1, 1, sk, IDX_D))
                  + mx.cumsum(uniform(base + 4, (1, 1, sk, IDX_D)) * 0.05, axis=2)).astype(dtype)
            mx.eval(k, v, ik)
        return k, v, ik

    def queries(seed: int, n: int, rows: int, groups: int, dtype):
        """q [1, H, rows, D] and index queries [1, G, rows, IDX_D] from key stream n."""
        base = seed * 100_000 + 10 + 2 * n
        with mx.stream(mx.cpu):
            q = (uniform(base, (1, H, rows, D)) * 0.5).astype(dtype)
            iq = uniform(base + 1, (1, groups, rows, IDX_D)).astype(dtype)
            mx.eval(q, iq)
        return q, iq

    def attention(groups: int) -> Attention:
        args = ModelArgs(hidden_size=256, num_hidden_layers=1, num_attention_heads=H, num_key_value_heads=groups,
                         head_dim=D, rotary_dim=64, index_n_heads=groups, index_head_dim=IDX_D,
                         index_block_size=BLOCK, index_topk_blocks=TOPK, index_local_blocks=1,
                         sparse_attention_freq=[1])
        return Attention(args, 0)

    def reference(q, k, v, top_idx, offset: int, scale: float) -> np.ndarray:
        """float64 MSA attention [rows, H * D] from the selection's block ids."""
        Q = np.array(q.astype(mx.float32), dtype=np.float64)[0]
        K = np.array(k.astype(mx.float32), dtype=np.float64)[0]
        V = np.array(v.astype(mx.float32), dtype=np.float64)[0]
        G, rows, sk = K.shape[0], Q.shape[1], K.shape[1]
        hg = H // G
        pos = np.arange(sk)
        top = np.array(top_idx).astype(np.int64)[0]
        out = np.zeros((rows, H, D))
        with np.errstate(all="ignore"):
            for r in range(rows):
                for g in range(G):
                    allowed = (pos <= offset + r) & np.isin(pos // BLOCK, top[g, r])
                    s = Q[g * hg:(g + 1) * hg, r] @ K[g, allowed].T * scale
                    p = np.exp(s - s.max(-1, keepdims=True))
                    out[r, g * hg:(g + 1) * hg] = (p / p.sum(-1, keepdims=True)) @ V[g, allowed]
        if not np.isfinite(out).all():
            fail("float64 reference not finite")
        return out.reshape(rows, H * D)

    def rows_of(o) -> np.ndarray:
        a = np.array(o.astype(mx.float32), dtype=np.float64)[0]
        return a.transpose(1, 0, 2).reshape(a.shape[1], -1)

    def rel(a, ref):
        return np.linalg.norm(a - ref, axis=-1) / np.linalg.norm(ref, axis=-1)

    def run(attn, path, q, k, v, sel, mask):
        """(output, the path the router actually took) with VMLX_MINIMAX_MSA forced to ``path``."""
        if path == "kernel_sg":
            os.environ.pop(msa.ENV_PATH, None)
            return msa.kernel_attention(q, k, v, sel, attn.scale, _matrix_units=False), "kernel_sg"
        os.environ[msa.ENV_PATH] = path
        try:
            took = msa.choose_path(sel, q, k, mask, attn.n_kv)
            out = attn.attend(q, k, v, sel, mask, None)
            mx.eval(out)
            return out, took
        finally:
            os.environ.pop(msa.ENV_PATH, None)

    def evaluate(attn, paths, q, k, v, sel, offset, mask, tol, kind):
        if sel is None:
            fail(f"{kind}: no selection (every block visible)")
        top = np.array(sel.top_idx)
        if all(np.array_equal(top[0, 0], top[0, g]) for g in range(1, top.shape[1])):
            fail(f"{kind}: every KV group selected the same blocks")
        ref = reference(q, k, v, sel.top_idx, offset, attn.scale)
        os.environ.pop(msa.ENV_PATH, None)
        auto = msa.choose_path(sel, q, k, mask, attn.n_kv)
        entry = {"selection": digest(sel.top_idx), "auto": auto, "paths": {}}
        for path in paths:
            out, took = run(attn, path, q, k, v, sel, mask)
            err = rel(rows_of(out), ref)
            if not np.isfinite(err).all() or err.max() > tol:
                fail(f"{kind}: path {path} (took {took}) max rel error {err.max():.3e} > {tol}")
            entry["paths"][path] = {"took": took, "output": digest(out), "rel_median": float(np.median(err)),
                                    "rel_max": float(err.max())}
        return entry

    report = {"mode": mode, "matrix_units": None, "decode": [], "verify": [], "chunk": []}
    for dt in DTYPES:
        dtype = getattr(mx, dt)
        tol = BF16_TOL if dt == "bfloat16" else FP32_TOL if mode == "strict" else TF32_TOL
        for sk, groups in DECODE:
            seed = seed_of("decode", sk, groups)
            attn = attention(groups)
            k, v, ik = context(seed, sk, groups, dtype)
            rng = np.random.default_rng(seed)
            extra = rng.integers(TOPK * BLOCK, sk - 3, 6).tolist() if sk - 3 > TOPK * BLOCK else []
            positions = sorted(p for p in {sk - 1, sk - 2, sk - 3, *extra} if p >= TOPK * BLOCK)
            case = {"dtype": dt, "sk": sk, "groups": groups, "seed": seed, "positions": [],
                    "inputs": {"k": digest(k), "v": digest(v), "ik": digest(ik)}}
            for p in positions:
                q, iq = queries(seed, p, 1, groups, dtype)
                sel = attn.indexer.select(iq, ik[:, :, :p + 1], p)
                e = evaluate(attn, ("dense", "decode", "gather"), q, k[:, :, :p + 1], v[:, :, :p + 1], sel, p, None,
                             tol, f"decode {dt} G{groups} Sk{sk} p{p}")
                e.update({"position": p, "q": digest(q), "iq": digest(iq)})
                case["positions"].append(e)
            report["decode"].append(case)
            print(f"[{mode}] decode {dt} G={groups} Sk={sk}: {len(positions)} positions", flush=True)
        for sk, groups, width in VERIFY:
            seed = seed_of("verify", sk, groups)
            attn = attention(groups)
            k, v, ik = context(seed, sk, groups, dtype)
            q, iq = queries(seed, width, width, groups, dtype)
            sel = attn.indexer.select(iq, ik, sk - width)
            e = evaluate(attn, ("dense", "decode", "gather"), q, k, v, sel, sk - width, None, tol,
                         f"verify {dt} G{groups} Sk{sk} w{width}")
            e.update({"dtype": dt, "sk": sk, "groups": groups, "width": width, "seed": seed,
                      "inputs": {"k": digest(k), "v": digest(v), "ik": digest(ik), "q": digest(q),
                                 "iq": digest(iq)}})
            report["verify"].append(e)
            print(f"[{mode}] verify {dt} G={groups} Sk={sk} rows={width}: auto {e['auto']}", flush=True)
        for sk, groups, rows in CHUNK:
            seed = seed_of("chunk", sk, groups, rows)
            attn = attention(groups)
            k, v, ik = context(seed, sk, groups, dtype)
            q, iq = queries(seed, 0, rows, groups, dtype)
            sel = attn.indexer.select(iq, ik, sk - rows)
            paths = ("dense", "kernel") + (("kernel_sg",) if dt == "bfloat16" else ())
            e = evaluate(attn, paths, q, k, v, sel, sk - rows, "causal", tol, f"chunk {dt} G{groups} Sk{sk} r{rows}")
            e.update({"dtype": dt, "sk": sk, "groups": groups, "rows": rows, "seed": seed,
                      "inputs": {"k": digest(k), "v": digest(v), "ik": digest(ik), "q": digest(q),
                                 "iq": digest(iq)}})
            report["chunk"].append(e)
            print(f"[{mode}] chunk {dt} G={groups} Sk={sk} rows={rows}: auto {e['auto']}", flush=True)
    report["matrix_units"] = bool(msa.nax_available())
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
    for kind in ("decode", "verify", "chunk"):
        for s, t in zip(modes["strict"][kind], modes["tf32"][kind]):
            if s["inputs"] != t["inputs"]:
                fail(f"{kind}: inputs differ between the TF32 modes (they come from the PRNG only)")
    src = a.vmlx / "vmlx_engine/models/minimax_m3/msa.py"
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    fixture = {
        "generator": "scripts/minimax_m3_fixtures/generate_msa_block_sparse_fixtures.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "vmlx_commit": a.vmlx_commit,
        "msa_py_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "mlx": mx.__version__,
        "python": platform.python_version(),
        "gpu": str(info.get("architecture", "")),
        "geometry": {"heads": H, "head_dim": D, "index_head_dim": IDX_D, "block": BLOCK, "topk": TOPK, "local": 1},
        "tolerance_vs_float64": {"float32": FP32_TOL, "float32_tf32": TF32_TOL, "bfloat16": BF16_TOL},
        "modes": modes,
    }
    a.output.write_text(json.dumps(fixture, indent=None, separators=(",", ":")))
    n = sum(len(c["positions"]) for c in modes["strict"]["decode"])
    print(f"wrote {a.output} ({a.output.stat().st_size} bytes): {n} decode positions, "
          f"{len(modes['strict']['verify'])} verify, {len(modes['strict']['chunk'])} chunk cases; "
          f"matrix units {modes['strict']['matrix_units']}")


if __name__ == "__main__":
    main()
