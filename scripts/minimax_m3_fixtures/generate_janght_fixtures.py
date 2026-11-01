#!/usr/bin/env python3
"""Golden bundle + logits for the Swift MiniMax-M3.1 JANGHT load (`Libraries/MLXLLM/MiniMaxM3JANGHPreparation.swift`).

A tiny synthetic `minimax_m3_vl` JANGHT bundle is written the way the M3.1 assembler writes one, and run through the
vMLX Python runtime's own JANG entry point (`vmlx_engine.utils.jang_loader.load_jang_model`, the route a bundle with
jang_config.json takes). Nothing here reads production weights.

Bundle (`m31-janght/`):
  - tensor names and per-module `quantization` keys = LIVE module paths (no `language_model.` prefix);
  - trunk linears mixed per module: MLX `mxfp8` and affine 4 / 6 / 8 (g64); embed affine 8, lm_head affine 6;
    routers F32 unquantized with `e_score_correction_bias`; norms BF16;
  - routed experts (layers 1-2): JANGT trellis units `{"mode": "jangt", "kbits": K}` with layer-wide gate/up outlier
    columns, per-expert down outlier columns (one -1 padding slot) and down column scales; two widths per layer;
  - top-level `jangt` header; jang_config.json `format: jangtq2`, `format_version: 2`, `format_name: JANGHT`;
  - BF16 vision tensors under their source names (a text-only load drops them);
  - `model.safetensors.index.json` (the M3.1 assembler always writes one).
The MSA geometry is M3.1's: 128-key blocks, top-16 blocks, local block 1, one indexer head per KV group (2 groups).

Reference (`janght-reference.safetensors` + `.json`), teacher-forced token ids:
  - prefill of PREFILL tokens (not a multiple of 128 / 64 / 8; 23 key blocks, so every query past token 2047 keeps
    16 of its blocks per KV group) -> logits at ROWS;
  - DECODE one-token steps on the prefill cache -> every step's logits;
  - prefill in CHUNK-token chunks through the cache -> logits at ROWS.
  The Python JANGT runtime sends a routed batch of >= 72 (token, slot) pairs to its sorted NAX prefill tiles; vmlx-swift
  does so from 512 (its measured crossover). These goldens are generated with the Python threshold set to 512 so that
  every batch runs the same kernel family in both engines: prefill (11412 pairs) on the NAX tiles, chunks (404 pairs)
  and decode steps (4 pairs) on the decode kernels. The chunked logits with Python's default threshold (NAX tiles for
  the chunks) are recorded too, with their distance to the threshold-matched ones: that is the noise floor between the
  two kernel families.
  - `deq_*`: the same model with plain float experts dequantized from the JANGT units into the raw input basis
    (rotation, signs, outlier columns, column scales folded in) run through M3's own SwiGLU-OAI: a reference that
    shares no kernel with the runtime.

    MLX_ENABLE_TF32=0 PYTHONPATH=<vmlx checkout> <python with mlx-lm 0.31.3> \\
        scripts/minimax_m3_fixtures/generate_janght_fixtures.py --vmlx <vmlx checkout> --vmlx-commit <sha> \\
        --output <empty directory>

Every cross-check below fails closed (the script exits non-zero and writes nothing usable):
  - the Python load gives every module exactly its entry (mode / bits / group size / kbits), binds every bundle tensor,
    and installs SwiGLU-OAI (alpha, limit) on every routed stack;
  - logits finite and non-zero, the selection is sparse (blocks > top-k) at the prefill length;
  - decode and chunked prefill agree with full prefill at the bf16 noise floor;
  - the JANGT runtime matches the dequantized reference at the noise floor, and both SiLU controls (plain and clamped at
    the same limit) miss it by a wide margin.
"""

from __future__ import annotations

import os

os.environ["MLX_ENABLE_TF32"] = "0"  # before MLX initializes: fp32 matmuls (outlier columns, routing sum) strict

import argparse
import hashlib
import json
import platform
import struct
import sys
from pathlib import Path

import numpy as np

D, I, SH, DENSE, E, TOPK, VOCAB = 256, 128, 128, 256, 8, 4, 256
HEADS, KVH, HD, ROT, LAYERS = 4, 2, 64, 32, 3
ALPHA, LIMIT = 1.702, 7.0
BLOCK, TOPK_BLOCKS, LOCAL = 128, 16, 1                 # M3.1's sparse_attention_config
PREFILL, DECODE, CHUNK = 2853, 5, 101                 # 2853 = 22 * 128 + 37; 101 * 4 = 404 routed pairs < 512
SWIFT_SORT_THRESHOLD = 512                             # JANGTRoutedExpertLayer.sortThreshold
ROWS = sorted(set(range(0, PREFILL, 61)) | set(range(PREFILL - 40, PREFILL)))
WIDTHS = {1: (2, 2.5), 2: (1.5, 2)}                    # MoE layer -> (gate/up K, down K)
GU_COLS = {1: [3, 100, 201], 2: [17, 250]}             # layer-wide gate/up outlier columns
DN_COLS = 2                                            # per-expert down outlier slots (expert 1 pads one with -1)

MX8 = {"mode": "mxfp8", "group_size": 32, "bits": 8}


def A(bits):
    return {"group_size": 64, "bits": bits}


TRUNK = {
    "model.embed_tokens": A(8), "lm_head": A(6),
    "model.layers.0.self_attn.q_proj": MX8, "model.layers.0.self_attn.o_proj": MX8,
    "model.layers.0.self_attn.k_proj": A(6), "model.layers.0.self_attn.v_proj": A(6),
    "model.layers.0.self_attn.indexer.index_q_proj": MX8, "model.layers.0.self_attn.indexer.index_k_proj": A(8),
    "model.layers.0.mlp.gate_proj": MX8, "model.layers.0.mlp.up_proj": MX8, "model.layers.0.mlp.down_proj": MX8,
    "model.layers.1.self_attn.q_proj": A(4), "model.layers.1.self_attn.k_proj": A(4),
    "model.layers.1.self_attn.v_proj": A(4), "model.layers.1.self_attn.o_proj": A(4),
    "model.layers.1.self_attn.indexer.index_q_proj": MX8, "model.layers.1.self_attn.indexer.index_k_proj": MX8,
    "model.layers.1.mlp.shared_experts.gate_proj": A(4), "model.layers.1.mlp.shared_experts.up_proj": A(4),
    "model.layers.1.mlp.shared_experts.down_proj": A(8),
    "model.layers.2.self_attn.q_proj": MX8, "model.layers.2.self_attn.k_proj": MX8,
    "model.layers.2.self_attn.v_proj": MX8, "model.layers.2.self_attn.o_proj": MX8,
    "model.layers.2.self_attn.indexer.index_q_proj": A(8), "model.layers.2.self_attn.indexer.index_k_proj": A(8),
    "model.layers.2.mlp.shared_experts.gate_proj": MX8, "model.layers.2.mlp.shared_experts.up_proj": MX8,
    "model.layers.2.mlp.shared_experts.down_proj": MX8,
}
VISION = {  # source names, BF16: a text-only load must drop them
    "vision_tower.patch_embedding.weight": (32, 3, 2, 4, 4),
    "multi_modal_projector.linear_1.weight": (64, 32),
    "patch_merge_mlp.0.weight": (32, 64),
}
SOURCES = ("models/minimax_m3/minimax_m3.py", "models/minimax_m3/msa.py", "jangt/switch.py", "jangt/moe_kernels.py",
           "jangt/prefill.py", "utils/jang_loader.py")


def fail(message):
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def config(jangt: bool) -> dict:
    tc = {
        "model_type": "minimax_m3", "hidden_size": D, "num_hidden_layers": LAYERS, "intermediate_size": I,
        "dense_intermediate_size": DENSE, "shared_intermediate_size": SH, "num_attention_heads": HEADS,
        "num_key_value_heads": KVH, "head_dim": HD, "rotary_dim": ROT, "rope_theta": 10_000.0, "rms_norm_eps": 1e-6,
        "vocab_size": VOCAB, "num_local_experts": E, "num_experts_per_tok": TOPK, "n_shared_experts": 1,
        "routed_scaling_factor": 2.0, "norm_topk_prob": True, "swiglu_alpha": ALPHA, "swiglu_limit": LIMIT,
        "moe_layer_freq": [0, 1, 1], "tie_word_embeddings": False,
        "sparse_attention_config": {
            "sparse_num_index_heads": KVH, "sparse_index_dim": HD, "sparse_block_size": BLOCK,
            "sparse_topk_blocks": TOPK_BLOCKS, "sparse_local_block": LOCAL, "sparse_attention_freq": [1, 1, 1],
            "sparse_disable_index_value": [1, 1, 1], "sparse_init_block": 0, "sparse_score_type": "max"},
    }
    cfg = {"model_type": "minimax_m3_vl", "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
           "text_config": tc, "image_token_index": 200025, "video_token_index": 200026,
           "quantization": {"group_size": 64, "bits": 8, **TRUNK}}
    if jangt:
        outl = {}
        for L, (kg, kd) in WIDTHS.items():
            key = f"model.layers.{L}.mlp.switch_mlp"
            cfg["quantization"].update({f"{key}.gate_proj": {"mode": "jangt", "kbits": kg},
                                        f"{key}.up_proj": {"mode": "jangt", "kbits": kg},
                                        f"{key}.down_proj": {"mode": "jangt", "kbits": kd}})
            outl.update({f"{key}.gate_proj": len(GU_COLS[L]), f"{key}.up_proj": len(GU_COLS[L]),
                         f"{key}.down_proj": DN_COLS})
        cfg["jangt"] = {"version": 1, "code": "v2_halfbits", "state_bits": 12, "rotation": "hadamard128-signs",
                        "outlier_columns": dict(sorted(outl.items()))}
    return cfg


def sylvester(n):
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def hblk(n_in):
    h = np.zeros((n_in, n_in))
    for b in range(0, n_in, 128):
        h[b:b + 128, b:b + 128] = sylvester(128)
    return h


def jt_unit(mx, VL, rng, n_in, n_out, K, scale, cols, col_scales, su=None):
    """JANGT tensors + the raw-basis float64 weights (E, n_out, n_in) they encode."""
    W = VL.words_per_row(n_in, 12, K)
    packed = rng.integers(0, 2 ** 32, size=(E, n_out, W), dtype=np.uint64).astype(np.uint32)
    sc = rng.uniform(*scale, size=(E, n_out)).astype(np.float32)
    if su is None:
        su = rng.choice([-1.0, 1.0], size=n_in).astype(np.float32)
    oc = np.asarray(cols, np.int32)
    wo = (rng.standard_normal((E, n_out, oc.shape[1])) * 0.05 * (oc >= 0)[:, None, :]).astype(np.float16)
    cs = rng.uniform(0.5, 1.5, size=(E, n_in)).astype(np.float32) if col_scales else None
    t = {"jt_packed": packed, "jt_scale": sc, "jt_su": su, "jt_ocol": oc, "jt_wout": wo}
    if cs is not None:
        t["jt_cs"] = cs
    hb, raw = hblk(n_in), np.zeros((E, n_out, n_in))
    for e in range(E):
        st = VL.unpack_states(packed[e], n_in, 12, K)
        w_rot = np.array(VL.values_of(mx.array(st), 1.0, 12, K)).astype(np.float64) * sc[e][:, None]
        keep = np.ones(n_in)
        keep[oc[e][oc[e] >= 0]] = 0.0
        s = su.astype(np.float64) * keep * (cs[e] if cs is not None else 1.0)
        with np.errstate(all="ignore"):
            raw[e] = (w_rot @ hb) * s[None, :]
        for j, c in enumerate(oc[e]):
            if c >= 0:
                raw[e][:, c] += wo[e, :, j].astype(np.float64)
    if not np.isfinite(raw).all():
        fail("non-finite dequantized expert weights")
    return t, raw


def write_aligned(mx, path: Path, tensors: dict):
    """safetensors with every payload aligned to its element size (the M3.1 assembler's layout: payloads by
    descending element size, header padded to 16 bytes). mx.save_safetensors does not align, and vMLX's loader rewrites
    such a shard in place before loading it."""
    names = {mx.bfloat16: "BF16", mx.float16: "F16", mx.float32: "F32", mx.uint32: "U32", mx.int32: "I32",
             mx.uint8: "U8"}
    arrays = {}
    for k, v in tensors.items():
        raw = v.view(mx.uint16) if v.dtype == mx.bfloat16 else v
        arrays[k] = (names[v.dtype], np.ascontiguousarray(np.array(raw)))
    items = sorted(arrays.items(), key=lambda kv: (-kv[1][1].itemsize, kv[0]))
    header, offset = {"__metadata__": {"format": "mlx"}}, 0
    for k, (dt, arr) in items:
        header[k] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [offset, offset + arr.nbytes]}
        offset += arr.nbytes
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-(8 + len(h)) % 16)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for _, (_, arr) in items:
            f.write(arr.tobytes())


def rel_rows(a, b):
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.abs(b).max() > 0):
        fail("non-finite or all-zero logits")
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmlx", required=True, type=Path)
    ap.add_argument("--vmlx-commit", required=True)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    out = a.output
    if out.exists() and any(out.iterdir()):
        fail(f"{out} is not empty")
    sys.path.insert(0, str(a.vmlx))

    import mlx.core as mx
    import mlx.nn as nn
    import mlx_lm

    from vmlx_engine.jangh import kernels as JK
    from vmlx_engine.jangt import switch as SW
    from vmlx_engine.jangt import v2l as VL
    from vmlx_engine.models.minimax_m3 import minimax_m3 as M
    from vmlx_engine.utils import jang_loader

    probe = mx.random.normal((64, 64), key=mx.random.key(0))
    p64 = np.array(probe).astype(np.float64)
    with np.errstate(all="ignore"):                      # Accelerate raises spurious FP flags; checked finite below
        exact = p64 @ p64
    tf32 = float(np.abs(np.array(probe @ probe).astype(np.float64) - exact).max())
    if not np.isfinite(tf32) or tf32 > 1e-4:
        fail(f"float32 matmul is not strict (max abs error {tf32}); MLX_ENABLE_TF32=0 did not take effect")
    if not JK.nax_available():
        fail("the NAX prefill tiles are unavailable here; the goldens need both kernel families")

    rng = np.random.default_rng(31)
    ref = M.Model(M.ModelArgs.from_dict(config(jangt=False)))
    flat = {}
    for k, v in nn.utils.tree_flatten(ref.parameters()):
        if k.endswith("norm.weight"):
            flat[k] = mx.array(rng.standard_normal(v.shape).astype(np.float32) * 0.1).astype(mx.bfloat16)
        elif k.endswith(".mlp.gate.weight") or k.endswith("e_score_correction_bias"):
            flat[k] = mx.array(rng.standard_normal(v.shape).astype(np.float32) * (0.3 if "gate" in k else 0.05))
        elif "switch_mlp" in k:
            continue
        else:
            std = 1.0 if "embed_tokens" in k else 0.06
            flat[k] = mx.array(rng.standard_normal(v.shape).astype(np.float32) * std).astype(mx.bfloat16)
    ref.load_weights(list(flat.items()), strict=False)
    nn.quantize(ref, class_predicate=lambda p, m: TRUNK.get(p, False))
    tensors = {k: v for k, v in nn.utils.tree_flatten(ref.parameters()) if "switch_mlp" not in k}
    for L, (kg, kd) in WIDTHS.items():
        gu_cols = np.tile(np.array(GU_COLS[L]), (E, 1))
        dn_cols = np.stack([rng.choice(I, size=DN_COLS, replace=False) for _ in range(E)])
        dn_cols[1, 1] = -1
        su_gu = rng.choice([-1.0, 1.0], size=D).astype(np.float32)
        sw = ref.model.layers[L].mlp.switch_mlp
        for p, n_in, n_out, K, scale, cols, cs, su in (("gate", D, I, kg, (0.15, 0.35), gu_cols, False, su_gu),
                                                       ("up", D, I, kg, (0.15, 0.35), gu_cols, False, su_gu),
                                                       ("down", I, D, kd, (0.01, 0.02), dn_cols, True, None)):
            t, raw = jt_unit(mx, VL, rng, n_in, n_out, K, scale, cols, cs, su)
            for leaf, v in t.items():
                tensors[f"model.layers.{L}.mlp.switch_mlp.{p}_proj.{leaf}"] = mx.array(v)
            getattr(sw, f"{p}_proj").weight = mx.array(raw.astype(np.float32))
    mx.eval(ref.parameters())
    bundle = dict(tensors)
    for k, shape in VISION.items():
        bundle[k] = mx.array(rng.standard_normal(shape).astype(np.float32) * 0.02).astype(mx.bfloat16)

    cfg = config(jangt=True)
    d = out / "m31-janght"
    d.mkdir(parents=True)
    shard = "model-00001-of-00001.safetensors"
    write_aligned(mx, d / shard, bundle)
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(v.nbytes for v in bundle.values())},
         "weight_map": {k: shard for k in sorted(bundle)}}, indent=1))
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    (d / "jang_config.json").write_text(json.dumps(
        {"format": "jangtq2", "format_version": 2, "format_name": "JANGHT", "model_name": "tiny-m31-janght"}, indent=1))

    # ---- the Python runtime's JANG entry point (format jangtq2 -> JANG v2 loader)
    import mlx_lm.utils
    mlx_lm.utils.load_tokenizer = lambda *args, **kw: object()     # the fixture bundle ships no tokenizer
    before = hashlib.sha256((d / shard).read_bytes()).hexdigest()
    model, _ = jang_loader.load_jang_model(d)
    if hashlib.sha256((d / shard).read_bytes()).hexdigest() != before:
        fail("the Python loader rewrote the shard; the fixture must be loaded as written")
    got_quant = {}
    for path, m in model.named_modules():
        if isinstance(m, SW.JTSwitchLinear):
            got_quant[path] = ("jangt", float(m.kbits))
        elif isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            got_quant[path] = ("mlx", m.mode, m.bits, m.group_size)
    want_quant = {k: (("jangt", float(v["kbits"])) if v.get("mode") == "jangt"
                      else ("mlx", v.get("mode", "affine"), v["bits"], v["group_size"]))
                  for k, v in cfg["quantization"].items() if isinstance(v, dict)}
    if got_quant != want_quant:
        fail(f"Python load: module quantization differs from the plan: {set(got_quant) ^ set(want_quant)}")
    params = dict(nn.utils.tree_flatten(model.parameters()))
    if set(params) != set(tensors):
        fail(f"Python load: parameters differ from the bundle text tensors: {set(params) ^ set(tensors)}")
    for k, v in tensors.items():
        if params[k].dtype != v.dtype or not mx.array_equal(params[k], v).item():
            fail(f"Python load: {k} is not the bundle tensor")
    for L in WIDTHS:
        sw = model.model.layers[L].mlp.switch_mlp
        if not isinstance(sw, SW.JTMixedSwitchGLU) or (sw.act, sw.alpha, sw.limit) != ("oai", ALPHA, LIMIT):
            fail(f"layer {L}: routed experts are not JANGT with SwiGLU-OAI")

    tokens = np.random.default_rng(7).integers(0, VOCAB, size=PREFILL + DECODE).astype(np.int32)
    rows = np.array(ROWS)

    def prefill(m):
        return np.array(m(mx.array(tokens[None, :PREFILL]))[0].astype(mx.float32))

    def decode(m):
        cache = m.make_cache()
        mx.eval(m(mx.array(tokens[None, :PREFILL]), cache=cache))
        steps = [m(mx.array(tokens[None, t:t + 1]), cache=cache)[0].astype(mx.float32)
                 for t in range(PREFILL, PREFILL + DECODE)]
        return np.array(mx.concatenate(steps, axis=0))

    def chunked(m):
        cache = m.make_cache()
        parts = [m(mx.array(tokens[None, s:min(s + CHUNK, PREFILL)]), cache=cache)[0].astype(mx.float32)
                 for s in range(0, PREFILL, CHUNK)]
        return np.array(mx.concatenate(parts, axis=0))

    default_threshold = SW.SORT_THRESHOLD
    SW.SORT_THRESHOLD = SWIFT_SORT_THRESHOLD
    py_prefill, py_decode, py_chunked = prefill(model), decode(model), chunked(model)
    SW.SORT_THRESHOLD = default_threshold
    py_chunked_default = chunked(model)
    deq_prefill, deq_decode = prefill(ref), decode(ref)

    checks = {}

    def summary(err):
        return {"median": float(np.median(err)), "p90": float(np.quantile(err, 0.9)), "max": float(err.max())}

    # Fail-closed numerics.
    checks["chunked_vs_prefill"] = summary(rel_rows(py_chunked[rows], py_prefill[rows]))
    checks["chunked_python_default_vs_matched"] = summary(rel_rows(py_chunked_default[rows], py_chunked[rows]))
    checks["runtime_vs_dequant_prefill"] = summary(rel_rows(py_prefill[rows], deq_prefill[rows]))
    checks["runtime_vs_dequant_decode"] = summary(rel_rows(py_decode, deq_decode))
    tol = 3e-2
    for k in ("chunked_vs_prefill", "runtime_vs_dequant_prefill", "runtime_vs_dequant_decode"):
        if checks[k]["median"] > tol:
            fail(f"{k}: median {checks[k]['median']} > {tol}")

    class ClampedSilu(nn.Module):
        def __call__(self, x_up, x_gate):
            return nn.silu(mx.minimum(x_gate, LIMIT)) * mx.clip(x_up, -LIMIT, LIMIT)

    class PlainSilu(nn.Module):
        def __call__(self, x_up, x_gate):
            return nn.silu(x_gate) * x_up

    for name, act in (("control_clamped_silu", ClampedSilu()), ("control_plain_silu", PlainSilu())):
        for L in WIDTHS:
            ref.model.layers[L].mlp.switch_mlp.activation = act
        err = rel_rows(prefill(ref)[rows], py_prefill[rows])
        checks[name] = summary(err) | {"min": float(err.min())}
        if np.median(err) < 5 * tol:
            fail(f"{name}: median {np.median(err)} is within 5x the tolerance; the control cannot fail")
    for L in WIDTHS:
        ref.model.layers[L].mlp.switch_mlp.activation = M.SwiGLU(ALPHA, LIMIT)

    n_blocks = -(-PREFILL // BLOCK)
    if n_blocks <= TOPK_BLOCKS:
        fail("the prefill does not exceed top-k blocks; the MSA selection would be trivial")
    checks["msa"] = {"key_blocks_at_prefill": n_blocks, "topk_blocks": TOPK_BLOCKS,
                     "sparse_query_positions": PREFILL - TOPK_BLOCKS * BLOCK}

    ref_tensors = {
        "tokens": mx.array(tokens), "rows": mx.array(rows.astype(np.int32)),
        "prefill": mx.array(py_prefill[rows]), "decode": mx.array(py_decode), "chunked": mx.array(py_chunked[rows]),
        "chunked_python_default": mx.array(py_chunked_default[rows]),
        "deq_prefill": mx.array(deq_prefill[rows]), "deq_decode": mx.array(deq_decode),
    }
    mx.save_safetensors(str(out / "janght-reference.safetensors"), ref_tensors)
    me = Path(__file__).resolve()
    (out / "janght-reference.json").write_text(json.dumps({
        "generator": "scripts/minimax_m3_fixtures/generate_janght_fixtures.py",
        "generator_sha256": hashlib.sha256(me.read_bytes()).hexdigest(),
        "vmlx_commit": a.vmlx_commit,
        "reference_sha256": {s: hashlib.sha256((a.vmlx / "vmlx_engine" / s).read_bytes()).hexdigest()
                             for s in SOURCES},
        "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "machine": platform.machine(),
        "float32_matmul": f"MLX_ENABLE_TF32=0 (64x64 probe max abs error {tf32:.2g})",
        "python_entry": "vmlx_engine.utils.jang_loader.load_jang_model",
        "python_sort_threshold": {"default": default_threshold, "goldens": SWIFT_SORT_THRESHOLD},
        "shapes": {"prefill": PREFILL, "decode": DECODE, "chunk": CHUNK, "rows": len(ROWS),
                   "routed_pairs": {"prefill": PREFILL * TOPK, "chunk": CHUNK * TOPK, "decode": TOPK}},
        "checks": checks,
    }, indent=1))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
