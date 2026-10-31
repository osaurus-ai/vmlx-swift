#!/usr/bin/env python3
"""Golden digests for the Swift port of vMLX's MiniMax-M3 fused attention (K1-K5, `MiniMaxM3FusedAttention`).

vMLX (`vmlx_engine/models/minimax_m3/fused_attn.py` + `kernels/attn_*.metal`) runs one decoder layer's attention half
for 1 .. 16 rows in five launches. The Swift port compiles the identical Metal sources (embedded byte for byte) with
the identical host constants, so every kernel output must be bit-identical between the two runtimes on identical
inputs. This script builds one M3.1-shaped attention layer (64 q heads x 128, 4 KV heads, partial RoPE 64, theta 1e7,
index 4 x 128 / 1 x 128, blocks of 128, top-16, local 1; q / k / v / o affine 4-bit g64, index MXFP8; hidden 1024)
and a sparse cache at several context lengths from MLX's counter-based PRNG on the CPU (U[-2, 2) with explicit keys,
so the Swift test regenerates exactly the same tensors and checks their digests first), runs the five kernels in the
host order Swift uses (K2 with INPLACE = 0: the new rows come back as outputs and are appended by the host), and
records SHA-256 digests (raw bytes, row-major; bfloat16 as its uint16 bits) of every input and output, plus
`attention_block`'s own result (Python's host, in-place appends) to prove the two host orders agree.

    PYTHONPATH=<vmlx checkout> <python with mlx> scripts/minimax_m3_fixtures/generate_fused_attn_fixtures.py \\
        --vmlx-commit <sha> --output Tests/MLXLMTests/Fixtures/MiniMaxM3/fused-attn.json

Cases (context lengths not multiples of 128 / 256): rows 1 / 5 at 700 keys (dense: every block visible), 16 rows
ending past 2,048 (dense and sparse rows in one launch), 1 / 5 / 16 rows at 3,001 and 8 rows at 9,000 (sparse).
Fails closed: Python's host and the kernel-by-kernel order give the same block output, the output is not degenerate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys

import numpy as np

HIDDEN, H, KV, HD, ROT, THETA, BLOCK, TOPK, EPS = 1024, 64, 4, 128, 64, 1e7, 128, 16, 1e-6
CASES = [(700, 1), (700, 5), (2040, 16), (3001, 1), (3001, 5), (3001, 16), (9000, 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmlx-commit", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    import mlx.core as mx
    import mlx.nn as nn

    from vmlx_engine.models.minimax_m3 import fused_attn as F
    from vmlx_engine.models.minimax_m3 import kernels as K
    from vmlx_engine.models.minimax_m3 import minimax_m3 as M
    from vmlx_engine.models.minimax_m3.cache import MiniMaxM3SparseCache

    def digest(x) -> str:
        x = mx.contiguous(x)
        mx.eval(x)
        arr = np.array(x.view(mx.uint16)) if x.dtype == mx.bfloat16 else np.array(x)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def uniform(n: int, shape):
        return mx.random.uniform(-2.0, 2.0, shape, key=mx.random.key(n), stream=mx.cpu)

    # ---- the layer (seeds 1 .. 20)
    tc = {"hidden_size": HIDDEN, "num_attention_heads": H, "num_key_value_heads": KV, "head_dim": HD,
          "rotary_dim": ROT, "rope_theta": THETA, "rms_norm_eps": EPS, "num_hidden_layers": 1,
          "sparse_attention_config": {"sparse_num_index_heads": KV, "sparse_index_dim": HD,
                                      "sparse_block_size": BLOCK, "sparse_topk_blocks": TOPK,
                                      "sparse_local_block": 1, "sparse_attention_freq": [1]}}
    args = M.ModelArgs.from_dict({"text_config": tc})

    class Layer:
        pass

    layer = Layer()
    layer.self_attn = attn = M.Attention(args, 0)
    layer.input_layernorm = M.GemmaRMSNorm(HIDDEN, EPS)
    ix = attn.indexer
    inputs = {}
    projections = [
        ("self_attn.q_proj", attn, "q_proj", H * HD, HIDDEN, "a4"),
        ("self_attn.k_proj", attn, "k_proj", KV * HD, HIDDEN, "a4"),
        ("self_attn.v_proj", attn, "v_proj", KV * HD, HIDDEN, "a4"),
        ("self_attn.o_proj", attn, "o_proj", HIDDEN, H * HD, "a4"),
        ("self_attn.indexer.index_q_proj", ix, "index_q_proj", KV * HD, HIDDEN, "mx8"),
        ("self_attn.indexer.index_k_proj", ix, "index_k_proj", HD, HIDDEN, "mx8"),
    ]
    for seed, (path, owner, name, o, i, fmt) in enumerate(projections, start=1):
        w = (uniform(seed, (o, i)) * (0.75 / math.sqrt(i))).astype(mx.bfloat16)
        if fmt == "mx8":
            m = nn.QuantizedLinear(i, o, bias=False, group_size=32, bits=8, mode="mxfp8")
            m.weight, m.scales = mx.quantize(w, group_size=32, bits=8, mode="mxfp8", stream=mx.cpu)
            parts = {"weight": m.weight, "scales": m.scales}
        else:
            m = nn.QuantizedLinear(i, o, bias=False, group_size=64, bits=4)
            m.weight, m.scales, m.biases = mx.quantize(w, group_size=64, bits=4, stream=mx.cpu)
            parts = {"weight": m.weight, "scales": m.scales, "biases": m.biases}
        setattr(owner, name, m)
        for k, v in parts.items():
            inputs[f"{path}.{k}"] = v
    norms = [("input_layernorm", layer.input_layernorm), ("self_attn.q_norm", attn.q_norm),
             ("self_attn.k_norm", attn.k_norm), ("self_attn.indexer.index_q_norm", ix.index_q_norm),
             ("self_attn.indexer.index_k_norm", ix.index_k_norm)]
    for seed, (path, nm) in enumerate(norms, start=11):
        nm.weight = (uniform(seed, nm.weight.shape) * 0.1).astype(mx.bfloat16)
        inputs[f"{path}.weight"] = nm.weight
    mx.eval(attn.parameters(), layer.input_layernorm.parameters())
    p = F.plan(layer)
    if p is None:
        sys.exit(f"FAIL: plan: {F.unsupported_reason(layer)}")

    def lanes(ctx: int, case: int):
        base = 1000 * case
        k = uniform(base + 1, (1, KV, ctx, HD)).astype(mx.bfloat16)
        v = uniform(base + 2, (1, KV, ctx, HD)).astype(mx.bfloat16)
        drift = mx.cumsum(uniform(base + 4, (1, 1, ctx, HD)) * 0.05, axis=2, stream=mx.cpu)
        ik = (uniform(base + 3, (1, 1, ctx, HD)) + drift).astype(mx.bfloat16)
        x = uniform(base + 5, (1, 16, HIDDEN)).astype(mx.bfloat16)
        mx.eval(k, v, ik, x)
        return k, v, ik, x

    cases = []
    for ci, (ctx, rows) in enumerate(CASES):
        k, v, ik, xall = lanes(ctx, ci + 1)
        x = xall[:, :rows]
        x2 = x.reshape(rows, HIDDEN)
        rec = {"ctx": ctx, "rows": rows, "inputs": {"k": digest(k), "v": digest(v), "ik": digest(ik), "x": digest(x)}}
        # kernel by kernel, Swift's host order (K2 INPLACE = 0, rows appended by the host)
        y = F.qkv(layer, p, x2)
        dts = (mx.bfloat16, mx.bfloat16, mx.bfloat16)
        nb = -(-(ctx + rows) // p.block)
        meta = mx.array([ctx, rows, nb, 0, 0, 0, 0, 0], dtype=mx.int32)
        q, iq, cnt, new = F.heads(layer, p, y, meta, (None, None, None), dts, False)
        kn, vn, ikn = new
        kl = mx.concatenate([k, kn.transpose(1, 0, 2)[None]], axis=2)
        vl = mx.concatenate([v, vn.transpose(1, 0, 2)[None]], axis=2)
        il = mx.concatenate([ik, ikn.transpose(1, 0, 2)[None]], axis=2)
        sc = F.scores(p, iq, il, meta, nb) if nb > p.keep else None
        o = F.msa(p, q, kl, vl, sc if sc is not None else F._dummy(), cnt, meta)
        h = F.oproj(layer, p, o.reshape(rows, p.H * p.D), x2)
        mx.eval(y, q, iq, kn, vn, ikn, o, h)
        out = {"y": digest(y), "q": digest(q), "iq": digest(iq), "kn": digest(kn), "vn": digest(vn),
               "ikn": digest(ikn), "o": digest(o), "h": digest(h)}
        if sc is not None:
            out["sc"] = digest(sc)
        # Python's own host (in-place appends into the lanes when it owns them) must agree
        c = MiniMaxM3SparseCache()
        c.state = (k, v, ik)
        hb = F.attention_block(layer, x, c)
        mx.eval(hb)
        if not bool(mx.array_equal(hb.reshape(rows, HIDDEN), h)):
            sys.exit(f"FAIL: case {ctx}/{rows}: attention_block != kernel-by-kernel order")
        lk, lv, li = (lane[..., ctx:ctx + rows, :] for lane in c.fused_lanes())
        if not (bool(mx.array_equal(lk, kn.transpose(1, 0, 2)[None])) and bool(mx.array_equal(li, ikn.transpose(1, 0, 2)[None]))):
            sys.exit(f"FAIL: case {ctx}/{rows}: appended lane rows differ")
        if float(mx.abs(h.astype(mx.float32) - x2.astype(mx.float32)).max()) == 0.0:
            sys.exit(f"FAIL: case {ctx}/{rows}: degenerate attention output")
        rec["outputs"] = out
        rec["sparse"] = sc is not None
        cases.append(rec)
        print(f"case ctx {ctx} rows {rows}: sparse {sc is not None}, h {out['h'][:12]}")

    doc = {
        "vmlxCommit": a.vmlx_commit,
        "mlx": mx.__version__,
        "platform": platform.platform(),
        "kernelSourcesSha256": {n: hashlib.sha256((K.DIR / f"{n}.metal").read_bytes()).hexdigest()
                                for n in ("attn_common", "attn_qkv", "attn_heads", "attn_scores", "attn_msa",
                                          "attn_oproj")},
        "geometry": {"hidden": HIDDEN, "heads": H, "kvHeads": KV, "headDim": HD, "rotaryDim": ROT, "ropeTheta": THETA,
                     "block": BLOCK, "topK": TOPK, "eps": EPS},
        "weights": {k: digest(v) for k, v in sorted(inputs.items())},
        "cases": cases,
    }
    with open(a.output, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print("wrote", a.output, len(cases), "cases")


if __name__ == "__main__":
    main()
