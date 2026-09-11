#!/usr/bin/env python3
"""Emit one native-MTP draft step as a Core ML program for the ANE probe.

The program is the Qwen3.5-family MTP head for ONE decode step, shaped for
the Neural Engine (channels-first [1, C, 1, rows] planes, rows = 32 tile,
fp16 datapath, int8 per-row weights via constexpr_affine_dequantize):

    fused  = fc(concat(rmsnorm(embed), rmsnorm(hidden)))
    a      = fused + o_proj(attn(q_norm(q(x)), k_norm(k(x)), v(x), KV window))
    h_out  = a + down(silu(gate(n)) * up(n))          # n = rmsnorm(a)
    token  = argmax(lm_head(rmsnorm(h_out)))

Weights are synthetic (timing + op-support probe, not parity against a real
bundle). The KV window, RoPE tables and the mask come in as input planes so
the compiled program is static; the host appends the new k/v row itself.

Usage:
    python3 mtp_head_mil.py --geometry qwen38-27b --window 1024 --out /tmp/head27b
    python3 mtp_head_mil.py --geometry flash-next --vocab 32768 --out /tmp/headfn32k
Then:
    aneprobe --mil /tmp/head27b/model.mil --weights /tmp/head27b/weight.bin \
             --in "$(cat /tmp/head27b/inputs.txt)" --out-bytes $(cat /tmp/head27b/out_bytes.txt)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

GEOMETRIES = {
    # hidden, heads, kv_heads, head_dim, intermediate, vocab
    "qwen38-27b": dict(hidden=5120, heads=24, kv_heads=4, head_dim=256, inter=17408, vocab=248320),
    "ornith-9b": dict(hidden=4096, heads=16, kv_heads=4, head_dim=256, inter=12288, vocab=248320),
    # Flash-Next MTP layer is MoE (512 experts, top-10, moe_inter 640); the
    # executed-bytes equivalent is a dense MLP of width 10*640. Routing is a
    # separate design problem (procedure bank per expert); this is the
    # bandwidth-equivalent stand-in for timing.
    "flash-next": dict(hidden=2560, heads=24, kv_heads=2, head_dim=256, inter=6400, vocab=248320),
}

ROWS = 32  # ANE fp16 plane pitch contract: rows * 2 bytes on the 64-byte grid
MAX_CONV_K = 4608  # single-conv K cliff (mlx-serve measured 2.6x past this)
MAX_CONV_N = 16384  # output-channel chunk for the lm_head (probe-chosen; tuned below)


def int8_weight(n, k, rng):
    q = rng.integers(-127, 128, size=(n, k, 1, 1), dtype=np.int8)
    s = np.full((n,), 1.0 / (127.0 * np.sqrt(k)), dtype=np.float16)
    return q, s


def linear(x, n, k, rng, name, k_chunks=None, n_chunks=None):
    """y = W x as 1x1 conv(s) on [1, k, 1, rows] -> [1, n, 1, rows]."""
    k_chunks = k_chunks or max(1, -(-k // MAX_CONV_K))
    while k % k_chunks:
        k_chunks += 1
    n_chunks = n_chunks or max(1, -(-n // MAX_CONV_N))
    while n % n_chunks:
        n_chunks += 1
    kc, nc = k // k_chunks, n // n_chunks
    outs = []
    for j in range(n_chunks):
        acc = None
        for i in range(k_chunks):
            q, s = int8_weight(nc, kc, rng)
            w = mb.constexpr_affine_dequantize(
                quantized_data=q, zero_point=np.int8(0), scale=s, axis=np.int32(0),
                name=f"{name}_w{j}_{i}")
            xin = x if k_chunks == 1 else mb.slice_by_size(
                x=x, begin=[0, i * kc, 0, 0], size=[1, kc, 1, ROWS], name=f"{name}_x{j}_{i}")
            p = mb.conv(x=xin, weight=w, name=f"{name}_p{j}_{i}")
            acc = p if acc is None else mb.add(x=acc, y=p, name=f"{name}_s{j}_{i}")
        outs.append(acc)
    return outs[0] if len(outs) == 1 else mb.concat(values=outs, axis=1, name=f"{name}_cat")


def linear_chunks(x, n, k, rng, name):
    """Like linear() but returns the list of [1, n/nc, 1, rows] chunk outputs."""
    n_chunks = max(1, -(-n // MAX_CONV_N))
    while n % n_chunks:
        n_chunks += 1
    nc = n // n_chunks
    return [linear(x, nc, k, rng, f"{name}{j}") for j in range(n_chunks)], nc


def chunk_argmax(logit_chunks, nc, name):
    """ANE-resident argmax: per chunk, (max value, index of max) as fp16.

    reduce_argmax is CPU-only under Core ML, so the index is recovered
    arithmetically: mask = clip(1 + 1024 * (x - max), 0, 1) is 1 exactly at
    the max (and at anything within ~1e-3 of it — below fp16 logit resolution
    at |x| >= 16, so effectively ties) and 0 elsewhere; max(mask * ramp) is
    then the (largest tied) index. Indices up to 16384 are exact in fp16.
    The host picks the winning chunk from NC pairs per row.
    """
    ramp = np.arange(nc, dtype=np.float16).reshape(1, nc, 1, 1)
    maxes, idxs = [], []
    for j, lg in enumerate(logit_chunks):
        m = mb.reduce_max(x=lg, axes=[1], keep_dims=True, name=f"{name}_max{j}")
        d = mb.sub(x=lg, y=m, name=f"{name}_dif_{j}")
        d = mb.mul(x=d, y=np.float16(1024.0), name=f"{name}_dscaled_{j}")
        d = mb.add(x=d, y=np.float16(1.0), name=f"{name}_dplus1_{j}")
        mask = mb.clip(x=d, alpha=np.float16(0.0), beta=np.float16(1.0), name=f"{name}_mask{j}")
        idx = mb.reduce_max(x=mb.mul(x=mask, y=ramp, name=f"{name}_mr{j}"), axes=[1], keep_dims=True,
                            name=f"{name}_idx{j}")
        maxes.append(m)
        idxs.append(idx)
    return maxes, idxs


def rmsnorm(x, c, name, eps=1e-6):
    sq = mb.mul(x=x, y=x, name=f"{name}_sq")
    mean = mb.reduce_mean(x=sq, axes=[1], keep_dims=True, name=f"{name}_mean")
    r = mb.rsqrt(x=mean, epsilon=np.float16(eps), name=f"{name}_rsqrt")
    n = mb.mul(x=x, y=r, name=f"{name}_n")
    w = np.ones((1, c, 1, 1), dtype=np.float16)
    return mb.mul(x=n, y=w, name=f"{name}_out")


def build(g, window, vocab, rng):
    H, NH, KVH, HD, INTER = g["hidden"], g["heads"], g["kv_heads"], g["head_dim"], g["inter"]
    G = NH // KVH
    shapes = [
        (1, H, 1, ROWS),           # a_hidden
        (1, H, 1, ROWS),           # b_embed
        (KVH, 1, HD, window),      # c_kcache  [KVH,1,HD,W]
        (KVH, 1, window, HD),      # d_vcache  [KVH,1,W,HD]
        (1, 1, 1, window),         # e_mask    (0 / -inf)
        (1, 1, HD, ROWS),          # f_cos
        (1, 1, HD, ROWS),          # g_sin
    ]
    specs = [mb.TensorSpec(shape=sh, dtype=types.fp16) for sh in shapes]

    @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
    def prog(a_hidden, b_embed, c_kcache, d_vcache, e_mask, f_cos, g_sin):
        fused_in = mb.concat(values=[rmsnorm(b_embed, H, "pre_e"), rmsnorm(a_hidden, H, "pre_h")],
                             axis=1, name="fused_in")
        x = linear(fused_in, H, 2 * H, rng, "fc")
        n1 = rmsnorm(x, H, "ln1")
        q = linear(n1, NH * HD, H, rng, "q")
        k = linear(n1, KVH * HD, H, rng, "k")
        v = linear(n1, KVH * HD, H, rng, "v")
        # per-head norm on q/k: [1, heads*HD, 1, rows] -> [1, heads, HD, rows]
        q4 = mb.reshape(x=q, shape=[1, NH, HD, ROWS], name="q4")
        k4 = mb.reshape(x=k, shape=[1, KVH, HD, ROWS], name="k4")

        def head_norm(t, nh, name):
            sq = mb.mul(x=t, y=t, name=f"{name}_sq")
            mean = mb.reduce_mean(x=sq, axes=[2], keep_dims=True, name=f"{name}_mean")
            r = mb.rsqrt(x=mean, epsilon=np.float16(1e-6), name=f"{name}_rsqrt")
            n = mb.mul(x=t, y=r, name=f"{name}_n")
            return mb.mul(x=n, y=np.ones((1, 1, HD, 1), dtype=np.float16), name=f"{name}_out")

        def rope(t, name):
            half = HD // 2
            x1 = mb.slice_by_size(x=t, begin=[0, 0, 0, 0], size=[-1, -1, half, -1], name=f"{name}_x1")
            x2 = mb.slice_by_size(x=t, begin=[0, 0, half, 0], size=[-1, -1, half, -1], name=f"{name}_x2")
            neg = mb.mul(x=x2, y=np.float16(-1.0), name=f"{name}_neg")
            rot = mb.concat(values=[neg, x1], axis=2, name=f"{name}_rot")
            return mb.add(x=mb.mul(x=t, y=f_cos, name=f"{name}_c"),
                          y=mb.mul(x=rot, y=g_sin, name=f"{name}_s"), name=f"{name}_out")

        qn = rope(head_norm(q4, NH, "qn"), "qr")     # [1, NH, HD, rows]
        kn = rope(head_norm(k4, KVH, "kn"), "kr")    # new k row(s): an output, host appends
        k_new = mb.reshape(x=kn, shape=[1, KVH * HD, 1, ROWS], name="k_new")
        # queries: [1, NH, HD, rows] -> [KVH, G, rows, HD]
        qg = mb.reshape(x=qn, shape=[KVH, G, HD, ROWS], name="qg")
        qt = mb.transpose(x=qg, perm=[0, 1, 3, 2], name="qt")
        scores = mb.matmul(x=qt, y=c_kcache, name="scores")            # [KVH, G, rows, W]
        scores = mb.mul(x=scores, y=np.float16(1.0 / np.sqrt(HD)), name="scores_scaled")
        scores = mb.add(x=scores, y=e_mask, name="scores_masked")
        probs = mb.softmax(x=scores, axis=-1, name="probs")
        ctx = mb.matmul(x=probs, y=d_vcache, name="ctx")               # [KVH, G, rows, HD]
        ctx_t = mb.transpose(x=ctx, perm=[0, 1, 3, 2], name="ctx_t")   # [KVH, G, HD, rows]
        ctx_flat = mb.reshape(x=ctx_t, shape=[1, NH * HD, 1, ROWS], name="ctx_flat")
        o = linear(ctx_flat, H, NH * HD, rng, "o")
        h1 = mb.add(x=x, y=o, name="h1")
        n2 = rmsnorm(h1, H, "ln2")
        gate = linear(n2, INTER, H, rng, "gate")
        up = linear(n2, INTER, H, rng, "up")
        act = mb.mul(x=mb.mul(x=gate, y=mb.sigmoid(x=gate, name="sig"), name="silu"), y=up, name="act")
        act = mb.mul(x=act, y=np.float16(1.0 / 16), name="act_s")   # fp16 accumulator headroom
        down = linear(act, H, INTER, rng, "down")
        down = mb.mul(x=down, y=np.float16(16.0), name="down_s")
        h_out = mb.add(x=h1, y=down, name="h_out")
        nf = rmsnorm(h_out, H, "lnf")
        chunks, nc = linear_chunks(nf, vocab, H, rng, "lm_head")        # NC x [1, nc, 1, rows]
        maxes, idxs = chunk_argmax(chunks, nc, "am")
        # [1, H + 2*KVH*HD + 2*NC, 1, rows]
        return mb.concat(values=[h_out, k_new, v] + maxes + idxs, axis=1, name="out")

    n_chunks = max(1, -(-vocab // MAX_CONV_N))
    while vocab % n_chunks:
        n_chunks += 1
    out_channels = H + 2 * KVH * HD + 2 * n_chunks
    return prog, shapes, out_channels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="qwen38-27b", choices=sorted(GEOMETRIES))
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=None, help="draft-vocab size (default: full)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--coremlc", default="/Applications/Xcode.app/Contents/Developer/usr/bin/coremlc")
    args = ap.parse_args()

    g = GEOMETRIES[args.geometry]
    vocab = args.vocab or g["vocab"]
    rng = np.random.default_rng(7)
    prog, shapes, out_channels = build(g, args.window, vocab, rng)

    os.makedirs(args.out, exist_ok=True)
    pkg = os.path.join(args.out, "head.mlpackage")
    shutil.rmtree(pkg, ignore_errors=True)
    model = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                       compute_units=ct.ComputeUnit.CPU_AND_NE,
                       pass_pipeline=ct.PassPipeline.EMPTY)
    model.save(pkg)
    compiled = os.path.join(args.out, "compiled")
    shutil.rmtree(compiled, ignore_errors=True)
    subprocess.run([args.coremlc, "compile", pkg, compiled], check=True, capture_output=True)
    mlmodelc = os.path.join(compiled, "head.mlmodelc")
    shutil.copy(os.path.join(mlmodelc, "model.mil"), os.path.join(args.out, "model.mil"))
    shutil.copy(os.path.join(mlmodelc, "weights", "weight.bin"), os.path.join(args.out, "weight.bin"))

    in_bytes = [int(np.prod(sh)) * 2 for sh in shapes]
    H = g["hidden"]
    out_bytes = out_channels * ROWS * 2
    with open(os.path.join(args.out, "inputs.txt"), "w") as f:
        f.write(",".join(str(b) for b in in_bytes))
    with open(os.path.join(args.out, "out_bytes.txt"), "w") as f:
        f.write(str(out_bytes))
    NH, KVH, HD, INTER = g["heads"], g["kv_heads"], g["head_dim"], g["inter"]
    params = (2 * H * H + H * NH * HD + 2 * H * KVH * HD + NH * HD * H + 3 * H * INTER + vocab * H)
    meta = dict(geometry=args.geometry, window=args.window, vocab=vocab, rows=ROWS, out_channels=out_channels,
                params=int(params), weight_bytes=os.path.getsize(os.path.join(args.out, "weight.bin")),
                inputs=in_bytes, out_bytes=out_bytes)
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
