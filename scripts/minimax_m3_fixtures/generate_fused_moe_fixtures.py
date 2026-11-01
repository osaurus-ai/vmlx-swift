#!/usr/bin/env python3
"""Golden digests for the Swift port of vMLX's MiniMax-M3 fused MoE block (K6 / K7 / R / K8, `MiniMaxM3FusedMoE`).

vMLX (`vmlx_engine/models/minimax_m3/fused_moe.py` + `kernels/m3_*.metal`) runs one sparse MoE layer's
`resid + block(x_normed)` for 1 .. 16 rows in four launches. The Swift port compiles the identical Metal sources with
the identical template arguments, so every kernel output must be bit-identical between the two runtimes on identical
inputs. This script builds the block the vMLX tests build (`tests/test_minimax_m3_fused_moe.py` `_block`: fp32 router
+ correction bias, JANGT routed experts with SwiGLU-OAI alpha 1.702 / limit 7, an affine-4 group-64 shared expert,
16 experts, top-4, hidden 1024, intermediate 512) from MLX's counter-based PRNG on the CPU (U[-2, 2) with explicit
keys; trellis words are the bit patterns of such floats), so the Swift test regenerates exactly the same tensors and
checks their digests first. Two geometries: JANGT gate/up 1.5 / down 2.5 bits with outlier columns (1 gate/up, 3 down
with one padded entry), and 2 / 3 bits without outliers.

Recorded per case (rows 1 / 2 / 5 / 16): digests (raw bytes, row-major; bfloat16 as its uint16 bits) of K6 (sig, xr),
K7 with the shared expert in-kernel (h, hs, inds, scores), R (hr), K8 (out with the shared expert from K7), the whole
`fused_moe_residual` with the shared expert forced in-kernel, and vMLX's unfused reference (whether it equals the fused
block: vMLX's claim).

    PYTHONPATH=<vmlx checkout> <python with mlx> scripts/minimax_m3_fixtures/generate_fused_moe_fixtures.py \\
        --vmlx-commit <sha> --output Tests/MLXLMTests/Fixtures/MiniMaxM3/fused-moe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys

import numpy as np

D, I, E, KS = 1024, 512, 16, 4
GEOMETRIES = [
    {"name": "k1.5-2.5-outliers", "kgu": 1.5, "kdn": 2.5, "nog": 1, "nod": 3, "seed": 100},
    {"name": "k2-3-plain", "kgu": 2.0, "kdn": 3.0, "nog": 0, "nod": 0, "seed": 200},
]
ROWS = [1, 2, 5, 16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmlx-commit", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    import mlx.core as mx
    import mlx.nn as nn

    from vmlx_engine.jangt import switch as SW
    from vmlx_engine.models.minimax_m3 import fused_moe as F
    from vmlx_engine.models.minimax_m3 import minimax_m3 as M

    def digest(x) -> str:
        x = mx.contiguous(x)
        mx.eval(x)
        arr = np.array(x.view(mx.uint16)) if x.dtype == mx.bfloat16 else np.array(x)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def uniform(n: int, shape):
        return mx.random.uniform(-2.0, 2.0, shape, key=mx.random.key(n), stream=mx.cpu)

    def block(g):
        s = g["seed"]
        args = M.ModelArgs(hidden_size=D, intermediate_size=I, shared_intermediate_size=I, num_local_experts=E,
                           num_experts_per_tok=KS)
        b = M.SparseMoeBlock(args)
        sw = SW.JTMixedSwitchGLU(D, I, E, {"mode": "jangt", "kbits": g["kgu"]}, {"mode": "jangt", "kbits": g["kdn"]},
                                 g["nog"], g["nod"], 7.0, "oai", 1.702)
        arrays = {}
        for pi, (name, p, n_in, n_out) in enumerate((("gate", sw.gate_proj, D, g["nog"]), ("up", sw.up_proj, D, g["nog"]),
                                                     ("down", sw.down_proj, I, g["nod"]))):
            base = s + 10 * pi
            p.jt_packed = uniform(base + 1, p.jt_packed.shape).view(mx.uint32)
            p.jt_scale = 0.035 + 0.0075 * uniform(base + 2, p.jt_scale.shape)
            p.jt_su = mx.where(uniform(base + 3, p.jt_su.shape) >= 0, 1.0, -1.0).astype(mx.float32)
            if n_out:
                if name == "down":
                    cols = np.array([[(e * 97 + o) % n_in for o in (11, 211, 389)][:n_out] for e in range(E)],
                                    np.int32)
                    cols[2, -1] = -1
                else:
                    cols = np.tile(np.array([37][:n_out], np.int32), (E, 1))
                p.jt_ocol = mx.array(cols)
                p.jt_wout = (0.0125 * uniform(base + 4, p.jt_wout.shape)).astype(mx.float16)
            if name == "down":
                p.jt_cs = 1.0 + 0.25 * uniform(base + 5, (E, n_in))
            for k in ("jt_packed", "jt_scale", "jt_su", "jt_ocol", "jt_wout") + (("jt_cs",) if name == "down" else ()):
                arrays[f"{name}.{k}"] = getattr(p, k)
        sw.up_proj.jt_su = sw.gate_proj.jt_su
        sw.up_proj.jt_ocol = sw.gate_proj.jt_ocol
        arrays["up.jt_su"], arrays["up.jt_ocol"] = sw.up_proj.jt_su, sw.up_proj.jt_ocol
        b.switch_mlp = sw
        b.gate.weight = 0.025 * uniform(s + 50, (E, D))
        b.e_score_correction_bias = 0.025 * uniform(s + 51, (E,))
        arrays["gate.weight"], arrays["bias"] = b.gate.weight, b.e_score_correction_bias
        for si, n in enumerate(("gate_proj", "up_proj", "down_proj")):
            lin = getattr(b.shared_experts, n)
            o, i = lin.weight.shape
            w = (uniform(s + 60 + si, (o, i)) * (0.75 / math.sqrt(i))).astype(mx.bfloat16)
            q = nn.QuantizedLinear(i, o, bias=False, group_size=64, bits=4)
            q.weight, q.scales, q.biases = mx.quantize(w, group_size=64, bits=4, stream=mx.cpu)
            setattr(b.shared_experts, n, q)
            for k in ("weight", "scales", "biases"):
                arrays[f"shared.{n}.{k}"] = getattr(q, k)
        mx.eval(b.parameters(), list(arrays.values()))
        return b, arrays

    geometries = []
    for g in GEOMETRIES:
        b, arrays = block(g)
        x_all = (uniform(g["seed"] + 70, (1, 16, D)) * 0.5).astype(mx.bfloat16)
        r_all = uniform(g["seed"] + 71, (1, 16, D)).astype(mx.bfloat16)
        mx.eval(x_all, r_all)
        cases = []
        for rows in ROWS:
            x, res = x_all[:, :rows], r_all[:, :rows]
            if not F.supported(b, x):
                sys.exit(f"FAIL: {g['name']} rows {rows}: block not supported by the fused kernels")
            x2 = mx.contiguous(x.reshape(-1, D))
            r2 = mx.contiguous(res.reshape(-1, D))
            sig, xr = F.router(x2, b.gate.weight, b.switch_mlp._unit()["su_gu"], rows)
            h, hs, inds, scores = F.gate_up(b, x2, sig, xr, True)
            hr = F.down_input(b, h, inds)
            out = F.down(b, hr, h, inds, scores, hs, r2, None)
            import os
            os.environ["VMLX_M3_FUSED_SHARED_ROWS"] = "16"
            full = F.fused_moe_residual(b, x, res)
            ref = F.unfused_moe_residual(b, x, res)
            mx.eval(sig, xr, h, hs, inds, scores, hr, out, full, ref)
            if not bool(mx.array_equal(full.reshape(rows, D), out)):
                sys.exit(f"FAIL: {g['name']} rows {rows}: fused_moe_residual != kernel-by-kernel")
            if float(mx.abs(out.astype(mx.float32) - r2.astype(mx.float32)).max()) == 0.0:
                sys.exit(f"FAIL: {g['name']} rows {rows}: degenerate block output")
            rec = {"rows": rows, "x": digest(x), "resid": digest(res),
                   "outputs": {"sig": digest(sig), "xr": digest(xr), "h": digest(h), "hs": digest(hs),
                               "inds": digest(inds), "scores": digest(scores), "hr": digest(hr), "out": digest(out)},
                   "unfused": digest(ref), "unfusedEqualsFused": bool(mx.array_equal(ref.reshape(rows, D), out)),
                   "inds": np.array(inds).tolist()}
            cases.append(rec)
            print(f"{g['name']} rows {rows}: unfused == fused {rec['unfusedEqualsFused']}, out {rec['outputs']['out'][:12]}")
        geometries.append({**g, "arrays": {k: digest(v) for k, v in sorted(arrays.items())}, "cases": cases})

    doc = {
        "vmlxCommit": a.vmlx_commit, "mlx": mx.__version__, "platform": platform.platform(),
        "geometry": {"hidden": D, "intermediate": I, "experts": E, "topK": KS},
        "geometries": geometries,
    }
    with open(a.output, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
