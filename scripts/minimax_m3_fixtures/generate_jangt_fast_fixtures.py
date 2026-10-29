#!/usr/bin/env python3
"""Golden digests for the Swift port of vMLX's fast JANGT decode kernels (`vmlx_engine/jangt/fast_decode.py`,
VMLX_JANGT_FAST=1) and of the fused M3 MoE block that runs them (K6 ROT=0, K7 ROUTED=0, fast routed experts, K8
ROUTED=0).

The fast kernels' Metal is GENERATED per shape (Python f-strings), so the Swift port carries the same generator and
must produce the same source strings: for every production configuration this records the kernel name, input /
output names and the SHA-256 of the source, both for the test geometries below and for MiniMax-M3.1's real shapes
(hidden 6144, expert intermediate 3072, JANGT K 1 / 1.5 / 2 / 2.5, outlier columns 0 / 8, 4 slots). Bit identity on
inputs: the fused-MoE fixture's block (`generate_fused_moe_fixtures.py`: same MLX PRNG keys, regenerated in Swift, input
digests checked first), rows 1 / 2 / 5 / 16, digests of every kernel output (raw bytes, row-major).

    PYTHONPATH=<vmlx checkout> <python with mlx> scripts/minimax_m3_fixtures/generate_jangt_fast_fixtures.py \\
        --vmlx-commit <sha> --output Tests/MLXLMTests/Fixtures/MiniMaxM3/jangt-fast.json [--sources-dir <dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_fused_moe_fixtures as GF  # noqa: E402

ROWS = [1, 2, 5, 16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmlx-commit", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sources-dir", default=None)
    a = ap.parse_args()

    import mlx.core as mx

    from vmlx_engine.jangt import fast_decode as FD
    from vmlx_engine.jangt import moe_kernels as MK
    from vmlx_engine.jangt import v2l as VL
    from vmlx_engine.models.minimax_m3 import fused_moe as F

    os.environ["VMLX_JANGT_FAST"] = "1"
    sources: dict = {}
    orig = mx.fast.metal_kernel

    def spy(name, input_names, output_names, source, **kw):
        sources[name] = {"inputs": list(input_names), "outputs": list(output_names),
                         "sha256": hashlib.sha256(source.encode()).hexdigest(), "length": len(source)}
        if a.sources_dir:
            os.makedirs(a.sources_dir, exist_ok=True)
            with open(os.path.join(a.sources_dir, name + ".metal"), "w") as f:
                f.write(source)
        return orig(name=name, input_names=input_names, output_names=output_names, source=source, **kw)

    mx.fast.metal_kernel = spy
    for fn in (FD._pack1_kernel, FD.gen_gu4, FD.gen_dn4, FD.gen_dn4c, FD._combine4):
        fn.cache_clear()

    def digest(x) -> str:
        x = mx.contiguous(x)
        mx.eval(x)
        arr = np.array(x.view(mx.uint16)) if x.dtype == mx.bfloat16 else np.array(x)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    # Real M3.1 shapes: sources only (no launch).
    real = []
    D, I, KS = 6144, 3072, 4
    for K in (1, 1.5, 2, 2.5):
        for n_out in (0, 8):
            W_gu = VL.words_per_row(D, 12, K)
            W_dn = VL.words_per_row(I, 12, K)
            names = []
            FD.gen_gu4(D, I, K, 12, n_out, W_gu, "bfloat16_t", "float", 7.0, "oai", 1.702, **FD.GU_CFG)
            names.append(f"jtf4_gu_k{D}_n{I}_K{MK._kname(K)}_o{n_out}_w{W_gu}_bfloat16_t_float"
                         f"{MK.act_tag('oai', 7.0, 1.702)}_O4_G2_S1_scalar")
            FD.gen_dn4c(I, D, K, 12, n_out, W_dn, "float", O=4, NOG=2, KS=KS, wload="scalar")
            names.append(f"jtf4c_dn_k{I}_n{D}_K{MK._kname(K)}_o{n_out}_w{W_dn}_float_O4_G2_s{KS}_scalar")
            FD.gen_dn4(I, D, K, 12, n_out, W_dn, "float", O=4, NOG=2, KSPLIT=1, wload="scalar")
            names.append(f"jtf4_dn_k{I}_n{D}_K{MK._kname(K)}_o{n_out}_w{W_dn}_float_O4_G2_S1_scalar")
            real.append({"K": K, "n_out": n_out, "W_gu": W_gu, "W_dn": W_dn, "kernels": names})
    for kin, xdt, per_row in ((D, "bfloat16_t", False), (I, "float", True)):
        FD._pack1_kernel(kin, xdt, per_row, True)
    FD._combine4(D, KS)
    real_sources = dict(sources)

    geometries = []
    for g in GF.GEOMETRIES:
        b, arrays = build_block(g, GF)
        U = b.switch_mlp._unit()
        sw = b.switch_mlp
        if not FD.supported(U, GF.D, GF.I):
            sys.exit(f"FAIL: {g['name']}: fast kernels do not cover the block")
        x_all = (GF_uniform(mx, g["seed"] + 70, (1, 16, GF.D)) * 0.5).astype(mx.bfloat16)
        r_all = GF_uniform(mx, g["seed"] + 71, (1, 16, GF.D)).astype(mx.bfloat16)
        mx.eval(x_all, r_all)
        cases = []
        for rows in ROWS:
            x, res = x_all[:, :rows], r_all[:, :rows]
            x2 = mx.contiguous(x.reshape(-1, GF.D))
            r2 = mx.contiguous(res.reshape(-1, GF.D))
            # Routing from the fused kernels (K6 ROT=0 + K7 ROUTED=0: inds / scores and the shared expert's hs).
            sig, _ = F.router(x2, b.gate.weight, U["su_gu"], rows, rot=False)
            h0, hs, inds, scores = F.gate_up(b, x2, sig, None, True, routed=False)
            pgu = FD.pack_rows1(x2, U["su_gu"])
            h = FD.routed_gate_up(x2, inds, U, sw.limit, sw.act, sw.alpha)
            pdn = FD.pack_rows1(h, U["sv_dn"], inds.reshape(-1))
            y = FD.routed_down(h, inds, scores, U)
            ry = FD.routed_fast(x2, inds, scores, U, sw.limit, sw.act, sw.alpha)
            out = F.down(b, None, h0, inds, scores, hs, r2, None, ry=ry)
            full = F.fused_moe_residual(b, x, res)
            # JTMixedSwitchGLU.routed with the lever on (the JANGT routed layer's decode route): routed_fast + cast.
            switch = sw.routed(x2.reshape(1, rows, GF.D), inds.reshape(1, rows, -1).astype(mx.int32),
                               scores.reshape(1, rows, -1))
            mx.eval(sig, hs, inds, scores, h, y, ry, out, full)
            if not bool(mx.array_equal(full.reshape(rows, GF.D), out)):
                sys.exit(f"FAIL: {g['name']} rows {rows}: fused_moe_residual (fast) != kernel-by-kernel")
            if not bool(mx.array_equal(y, ry)):
                sys.exit(f"FAIL: {g['name']} rows {rows}: routed_fast != gate_up + down")
            rec = {"rows": rows, "x": digest(x), "resid": digest(res),
                   "outputs": {"sig": digest(sig), "hs": digest(hs), "inds": digest(inds), "scores": digest(scores),
                               "pgu_xq": digest(pgu[0]), "pgu_xinv": digest(pgu[1]), "pgu_st": digest(pgu[2]),
                               "h": digest(h), "pdn_xq": digest(pdn[0]), "pdn_xinv": digest(pdn[1]),
                               "pdn_st": digest(pdn[2]), "ry": digest(ry), "out": digest(out)},
                   "inds": np.array(inds).tolist()}
            if switch is not None:
                mx.eval(switch)
                rec["outputs"]["switch"] = digest(switch.reshape(rows, GF.D))
            cases.append(rec)
            print(f"{g['name']} rows {rows}: ry {rec['outputs']['ry'][:12]} out {rec['outputs']['out'][:12]}")
        geometries.append({**g, "arrays": {k: digest(v) for k, v in sorted(arrays.items())}, "cases": cases})

    doc = {
        "vmlxCommit": a.vmlx_commit, "mlx": mx.__version__, "platform": platform.platform(),
        "geometry": {"hidden": GF.D, "intermediate": GF.I, "experts": GF.E, "topK": GF.KS},
        "geometries": geometries,
        "realShapes": real,
        "sources": dict(sorted(sources.items())),
        "realSourceNames": sorted(real_sources),
    }
    with open(a.output, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print("wrote", a.output, len(sources), "kernel sources")


def GF_uniform(mx, n, shape):
    return mx.random.uniform(-2.0, 2.0, shape, key=mx.random.key(n), stream=mx.cpu)


def build_block(g, GF):
    """The fused-MoE fixture's block (generate_fused_moe_fixtures.block), unchanged."""
    import math

    import mlx.core as mx
    import mlx.nn as nn

    from vmlx_engine.jangt import switch as SW
    from vmlx_engine.models.minimax_m3 import minimax_m3 as M

    def uniform(n, shape):
        return GF_uniform(mx, n, shape)

    D, I, E, KS = GF.D, GF.I, GF.E, GF.KS
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
                cols = np.array([[(e * 97 + o) % n_in for o in (11, 211, 389)][:n_out] for e in range(E)], np.int32)
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


if __name__ == "__main__":
    main()
