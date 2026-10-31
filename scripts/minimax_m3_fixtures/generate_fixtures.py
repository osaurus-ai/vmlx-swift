#!/usr/bin/env python3
"""Golden tensors for the Swift MiniMax-M3 / M3.1 text model (`Libraries/MLXLLM/Models/MiniMaxM3*.swift`).

Tiny synthetic models only: nothing here reads production weights. The reference is the vMLX Python implementation
(`vmlx_engine/models/minimax_m3/`: minimax_m3.py, cache.py, source_format.py), imported from `--vmlx`. Its file hashes
and the commit named by `--vmlx-commit` are written to `reference.json`.

MLX on M5 runs float32 matmuls as TF32 unless MLX_ENABLE_TF32=0 is set before MLX initializes. This script sets it,
and checks that it took effect, so every float32 golden is strict float32.

    PYTHONPATH=<vmlx checkout> python scripts/minimax_m3_fixtures/generate_fixtures.py \\
        --vmlx <vmlx checkout> --vmlx-commit <sha> --output <empty directory>

Generate into a separate directory and compare with `Tests/MLXLMTests/Fixtures/MiniMaxM3` before replacing anything.

Output:
  attention/        one sparse attention layer, its input, and an independent float64 NumPy reference
                    (output and per-KV-group block selection) written from the upstream MSA semantics
  m31-float/        5 layers, M3.1 layout (sparse attention on every layer, MoE from layer 3), float32,
                    checkpoint key names (`language_model.model.*`, `block_sparse_moe`, flat `self_attn.index_*`)
  m3-float/         the same shapes in the M3 layout (dense attention on layers 0-2)
  m31-source/       2 layers in the official stored formats: MXFP8 linears (F8_E4M3 weight + F8_E8M0
                    `weight_scale_inv`), NVFP4 routed experts per expert (U8 `weight_packed`, F8_E4M3
                    `weight_scale`, F32 `weight_global_scale` that differs per expert, unused `input_global_scale`),
                    BF16 embeddings / norms / lm_head and F32 router, plus vision and MTP tensors that a
                    text-only load must drop
  reference.safetensors  tokens, logits (prefill, decode through the cache, chunked prefill), per-layer key
                    selections, and the float-dequant reference of the source model
  reference.json    provenance and the measured agreement of every cross-check below

Cross-checks this script asserts before writing anything (it fails closed):
  - the attention module matches the NumPy reference, and the KV groups' selections differ somewhere;
  - in each float model, every sparse layer's selection equals the NumPy selection, at least one layer has groups
    that disagree, and decode / chunked prefill through the cache reproduce full prefill;
  - the source model, loaded as stored, matches a float model built from a spec dequant of the stored bytes, and
    the routed experts change its output (so an expert-layout error could not pass);
  - top-k margins (MSA blocks and routed experts) are recorded so a reader can see how far from a tie they are.
"""

from __future__ import annotations

import os

os.environ["MLX_ENABLE_TF32"] = "0"  # before MLX initializes

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

# Tiny shapes shared by the float models.
HID, HEADS, KV, HD, ROT = 32, 4, 2, 8, 4
INTER, SHARED, DENSE, EXPERTS, TOPK_E, VOCAB = 16, 16, 32, 4, 2, 64
BLOCK, TOPK_B, LOCAL = 4, 2, 1
LAYERS = 5
SEQ, PREFIX, CHUNK = 23, 15, 7  # 23 > TOPK_B * BLOCK and not a multiple of BLOCK
# Smallest allowed gap between the last kept and the first dropped block / expert score. Strict float32 consumers
# differ from these goldens by ~1e-6, so 0.01 keeps every top-k choice far from a tie for them. TF32 does not fit
# under any practical margin here: on this Mac it moved the float models' block scores by up to 0.043 (scores up to
# ~10), so a TF32 consumer cannot be expected to reproduce the selections; compare in strict float32 (CPU, or GPU with
# MLX_ENABLE_TF32=0).
MIN_MARGIN = 0.01

# Source-format model (shapes from the vMLX source-format test: MXFP8 needs inputs in groups of 32).
S_HID, S_HEADS, S_KV, S_HD, S_ROT, S_INTER, S_E, S_VOCAB = 64, 4, 2, 16, 8, 32, 4, 96
S_SEQ, S_PREFIX = 13, 9

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
SOURCES = ("minimax_m3.py", "cache.py", "source_format.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------- configs


def text_config(layers, sparse, moe, *, hidden, heads, kv, head_dim, rot, inter, shared, dense, experts, vocab):
    return {
        "model_type": "minimax_m3",
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "intermediate_size": inter,
        "dense_intermediate_size": dense,
        "shared_intermediate_size": shared,
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "head_dim": head_dim,
        "rotary_dim": rot,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "vocab_size": vocab,
        "num_local_experts": experts,
        "num_experts_per_tok": TOPK_E,
        "n_shared_experts": 1,
        "routed_scaling_factor": 2.0,
        "norm_topk_prob": True,
        "swiglu_alpha": 1.702,
        "swiglu_limit": 7.0,
        "moe_layer_freq": moe,
        "tie_word_embeddings": False,
        "sparse_attention_config": {
            "sparse_num_index_heads": kv,
            "sparse_index_dim": head_dim,
            "sparse_block_size": BLOCK,
            "sparse_topk_blocks": TOPK_B,
            "sparse_local_block": LOCAL,
            "sparse_attention_freq": sparse,
            "sparse_disable_index_value": [1] * layers,
            "sparse_init_block": 0,
            "sparse_score_type": "max",
        },
    }


def vl_config(text, **extra):
    return {
        "model_type": "minimax_m3_vl",
        "architectures": ["MiniMaxM3VLForConditionalGeneration"],
        "text_config": text,
        "image_token_index": 200025,
        "video_token_index": 200026,
        **extra,
    }


FLOAT_SHAPES = dict(hidden=HID, heads=HEADS, kv=KV, head_dim=HD, rot=ROT, inter=INTER, shared=SHARED, dense=DENSE,
                    experts=EXPERTS, vocab=VOCAB)
SOURCE_SHAPES = dict(hidden=S_HID, heads=S_HEADS, kv=S_KV, head_dim=S_HD, rot=S_ROT, inter=S_INTER, shared=S_INTER,
                     dense=S_HID, experts=S_E, vocab=S_VOCAB)


# ----------------------------------------------------------------------------------------------- numpy reference


def np_norm(x, w, eps=1e-6):
    x = x.astype(np.float64)
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * (1.0 + w)


def np_rope(x, pos, rot):  # x [S, heads, D]; the first `rot` dims rotate, half-split layout
    half = rot // 2
    inv = 10_000.0 ** (-np.arange(0, rot, 2) / rot)
    ang = pos[:, None, None] * inv[None, None, :]
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :half], x[..., half:rot]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s, x[..., rot:]], axis=-1)


def np_block_selection(iq, ik, S):
    """Upstream indexer for one sequence. iq [S, G, D], ik [S, D] (normed, roped).

    Returns (selected [G, S, n_blocks] bool, margin [G, S]): margin is the gap between the last kept and the first
    dropped block score (inf where every causally visible block is kept)."""
    G = iq.shape[1]
    nb = -(-S // BLOCK)
    causal = np.arange(S)[None, :] <= np.arange(S)[:, None]
    sel = np.zeros((G, S, nb), bool)
    margin = np.full((G, S), np.inf)
    for g in range(G):
        sc = np.where(causal, iq[:, g] @ ik.T, -np.inf)
        sc = np.pad(sc, ((0, 0), (0, nb * BLOCK - S)), constant_values=-np.inf)
        bs = sc.reshape(S, nb, BLOCK).max(-1)
        bs[np.arange(S), np.arange(S) // BLOCK] = np.inf  # the local block is always kept
        for i in range(S):
            valid = np.flatnonzero(bs[i] > -np.inf)
            order = valid[np.argsort(-bs[i, valid], kind="stable")]
            sel[g, i, order[:TOPK_B]] = True
            if len(order) > TOPK_B:
                margin[g, i] = bs[i, order[TOPK_B - 1]] - bs[i, order[TOPK_B]]
    if nb <= TOPK_B:
        sel[:] = True
    return sel, margin


def key_mask(sel, S):
    """Block selection [G, S, nb] -> attended keys [G, S, S] (selected and causal)."""
    causal = np.arange(S)[None, :] <= np.arange(S)[:, None]
    return np.repeat(sel, BLOCK, axis=2)[:, :, :S] & causal[None]


def np_attention(x, W, S):
    """Upstream MSA attention for one sequence (float64). Returns (output [S, HID], selection [G, S, nb])."""
    pos = np.arange(S, dtype=np.float64)
    q = np_rope(np_norm((x @ W["q"].T).reshape(S, HEADS, HD), W["qn"]), pos, ROT)
    k = np_rope(np_norm((x @ W["k"].T).reshape(S, KV, HD), W["kn"]), pos, ROT)
    v = (x @ W["v"].T).reshape(S, KV, HD)
    iq = np_rope(np_norm((x @ W["iq"].T).reshape(S, KV, HD), W["iqn"]), pos, ROT)
    ik = np_rope(np_norm((x @ W["ik"].T).reshape(S, 1, HD), W["ikn"]), pos, ROT)[:, 0]
    sel, _ = np_block_selection(iq, ik, S)
    keep = key_mask(sel, S)
    out = np.zeros((S, HEADS, HD))
    for h in range(HEADS):
        g = h // (HEADS // KV)
        logits = np.where(keep[g], (q[:, h] @ k[:, g].T) * HD ** -0.5, -np.inf)
        p = np.exp(logits - logits.max(-1, keepdims=True))
        out[:, h] = (p / p.sum(-1, keepdims=True)) @ v[:, g]
    return out.reshape(S, HEADS * HD) @ W["o"].T, sel


# ----------------------------------------------------------------------------------------------- stored formats


def e4m3_value(b):
    b = b.astype(np.int64)
    s, e, m = b >> 7, (b >> 3) & 15, b & 7
    v = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7))
    return np.where(s == 1, -v, v)


def mxfp8(rng, out_d, in_d):
    """e4m3 weight bytes + e8m0 scale per 32 inputs, and the float64 spec dequant."""
    sign = rng.integers(0, 2, (out_d, in_d)) << 7
    w = (sign | (rng.integers(5, 9, (out_d, in_d)) << 3) | rng.integers(0, 8, (out_d, in_d))).astype(np.uint8)
    s = rng.integers(121, 124, (out_d, in_d // 32)).astype(np.uint8)
    return w, s, e4m3_value(w) * np.repeat(2.0 ** (s.astype(np.float64) - 127), 32, axis=1)


def nvfp4(rng, out_d, in_d):
    """Two e2m1 codes per byte (low nibble first), e4m3 scale per 16, one global scale the weight is divided by."""
    packed = rng.integers(0, 256, (out_d, in_d // 2)).astype(np.uint8)
    scale = rng.integers(0x30, 0x48, (out_d, in_d // 16)).astype(np.uint8)
    g = float(rng.uniform(50, 200))
    codes = np.empty((out_d, in_d), np.uint8)
    codes[:, 0::2], codes[:, 1::2] = packed & 15, packed >> 4
    return packed, scale, g, E2M1[codes] * np.repeat(e4m3_value(scale), 16, axis=1) / g


def save_checkpoint(path: Path, tensors: dict):
    """Write a safetensors shard with the checkpoint's own dtypes.

    Values are (numpy array, dtype) with dtype one of F32, BF16, F8_E4M3, F8_E8M0, U8."""
    import torch
    from safetensors.torch import save_file

    out = {}
    for name, (value, dtype) in tensors.items():
        if dtype == "F32":
            out[name] = torch.from_numpy(np.ascontiguousarray(value, np.float32))
        elif dtype == "BF16":
            out[name] = torch.from_numpy(np.ascontiguousarray(value, np.float32)).to(torch.bfloat16)
        elif dtype in ("F8_E4M3", "F8_E8M0", "U8"):
            t = torch.from_numpy(np.ascontiguousarray(value, np.uint8))
            out[name] = t.view({"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu,
                                "U8": torch.uint8}[dtype])
        else:
            raise ValueError(dtype)
    save_file(out, str(path), metadata={"format": "pt"})


def safetensors_dtypes(path: Path) -> dict:
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n])
    return {k: v["dtype"] for k, v in header.items() if k != "__metadata__"}


# ----------------------------------------------------------------------------------------------- fixtures


class Recorder:
    """Captures what every sparse layer's indexer and every MoE router sees during one forward."""

    def __init__(self, mm):
        self.mm = mm
        self.indexer, self.router = [], []

    def __enter__(self):
        mm, rec = self.mm, self
        self._ix, self._moe = mm.Indexer.__call__, mm.SparseMoeBlock.__call__

        def ix(module, x, cache, offset):
            out = rec._ix(module, x, cache, offset)
            rec.indexer.append((module, x, offset, out))
            return out

        def moe(module, x):
            rec.router.append((module, x))
            return rec._moe(module, x)

        mm.Indexer.__call__, mm.SparseMoeBlock.__call__ = ix, moe
        return self

    def __exit__(self, *exc):
        self.mm.Indexer.__call__, self.mm.SparseMoeBlock.__call__ = self._ix, self._moe


def indexer_keys(module, x):
    """The module's own normed + roped indexer q / k for an uncached prefill (offset 0), as float64."""
    import mlx.core as mx

    B, S, _ = x.shape
    iq = module.index_q_norm(module.index_q_proj(x).reshape(B, S, module.nh, module.d)).transpose(0, 2, 1, 3)
    ik = module.index_k_norm(module.index_k_proj(x).reshape(B, S, 1, module.d)).transpose(0, 2, 1, 3)
    iq, ik = module.rope(iq, offset=0), module.rope(ik, offset=0)
    iq = np.array(iq.astype(mx.float32), np.float64)[0].transpose(1, 0, 2)  # [S, G, D]
    ik = np.array(ik.astype(mx.float32), np.float64)[0, 0]  # [S, D]
    return iq, ik


def router_margin(module, x):
    import mlx.core as mx

    scores = mx.sigmoid(module.gate(x).astype(mx.float32)) + module.e_score_correction_bias
    s = -np.sort(-np.array(scores, np.float64).reshape(-1, scores.shape[-1]), axis=-1)
    return float((s[:, TOPK_E - 1] - s[:, TOPK_E]).min())


def check_selection(rec, S, label, min_margin=0.0):
    """Every sparse layer's key selection equals the NumPy selection from the module's own indexer q / k.

    Returns None, before comparing anything, when some top-k choice is closer than `min_margin` to a tie: there the
    float32 module and the float64 reference may legitimately pick different blocks. Past that margin any difference
    is a defect and fails."""
    import mlx.core as mx

    layers = []
    for module, x, offset, bias in rec.indexer:
        assert offset == 0 and bias is not None, f"{label}: expected an uncached prefill past the top-k threshold"
        iq, ik = indexer_keys(module, x)
        sel, margin = np_block_selection(iq, ik, S)
        layers.append((np.array(bias.astype(mx.float32))[0] == 0, sel, float(margin.min())))
    margins = [m for _, _, m in layers]
    if min(margins) < min_margin:
        return None
    masks, disagree = [], []
    for got, sel, _ in layers:  # got: [G, S, S] attended keys
        assert (got == key_mask(sel, S)).all(), f"{label}: Python key selection differs from the NumPy selection"
        masks.append(got)
        disagree.append(int((sel[0] != sel[1]).any(-1).sum()))
    assert max(disagree) > 0, f"{label}: no layer has KV groups that disagree; the fixture would prove nothing"
    return np.stack(masks).astype(np.uint8), disagree, margins


def build_attention(mm, mx, out_dir: Path, reference: dict, report: dict):
    """One sparse attention layer (the vMLX per-group test's shapes and seeds) and its NumPy reference."""
    rng = np.random.default_rng(0)
    w = lambda *s: (rng.standard_normal(s) * 0.3).astype(np.float32)  # noqa: E731
    W = {"q": w(HEADS * HD, HID), "k": w(KV * HD, HID), "v": w(KV * HD, HID), "o": w(HID, HEADS * HD),
         "qn": w(HD), "kn": w(HD), "iq": w(KV * HD, HID), "ik": w(HD, HID), "iqn": w(HD), "ikn": w(HD)}
    x = np.random.default_rng(1).standard_normal((SEQ, HID)).astype(np.float32)
    ref, sel = np_attention(x.astype(np.float64), W, SEQ)
    disagree = int((sel[0] != sel[1]).any(-1).sum())
    assert -(-SEQ // BLOCK) > TOPK_B and disagree > 0, "attention: KV groups never disagree"

    config = vl_config(text_config(4, [1, 1, 1, 1], [0, 0, 0, 1], **FLOAT_SHAPES))
    args = mm.ModelArgs.from_dict(config)
    a = mm.Attention(args, 0)
    names = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj", "qn": "q_norm", "kn": "k_norm",
             "iq": "indexer.index_q_proj", "ik": "indexer.index_k_proj", "iqn": "indexer.index_q_norm",
             "ikn": "indexer.index_k_norm"}
    a.load_weights([(names[k] + ".weight", mx.array(v)) for k, v in W.items()], strict=True)
    got = np.array(a(mx.array(x)[None], mask="causal", cache=None))[0]
    err = float(np.abs(got - ref).max() / np.abs(ref).max())
    assert err < 1e-5, f"attention: vMLX module vs NumPy reference {err:.2e}"

    (out_dir / "attention").mkdir()
    (out_dir / "attention" / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for k, v in W.items():
        reference[f"attention.weight.{names[k]}.weight"] = v
    reference["attention.x"] = x[None]
    reference["attention.expected"] = ref.astype(np.float32)
    reference["attention.key_mask"] = key_mask(sel, SEQ).astype(np.uint8)
    report["attention"] = {"python_vs_numpy_rel": err, "group_disagreements": disagree,
                           "seq": SEQ, "block": BLOCK, "topk_blocks": TOPK_B}


def float_checkpoint(seed, sparse, moe):
    """Float32 weights under the checkpoint's names (routed experts already stacked as `switch_mlp`, the only float
    expert layout the vMLX sanitize accepts)."""
    rng = np.random.default_rng(seed)
    f = lambda *s: (rng.standard_normal(s) * 0.2).astype(np.float32)  # noqa: E731
    t = {"language_model.model.embed_tokens.weight": f(VOCAB, HID), "language_model.lm_head.weight": f(VOCAB, HID),
         "language_model.model.norm.weight": f(HID)}
    for L in range(LAYERS):
        p = f"language_model.model.layers.{L}."
        t[p + "input_layernorm.weight"], t[p + "post_attention_layernorm.weight"] = f(HID), f(HID)
        a = p + "self_attn."
        t[a + "q_proj.weight"], t[a + "k_proj.weight"] = f(HEADS * HD, HID), f(KV * HD, HID)
        t[a + "v_proj.weight"], t[a + "o_proj.weight"] = f(KV * HD, HID), f(HID, HEADS * HD)
        t[a + "q_norm.weight"], t[a + "k_norm.weight"] = f(HD), f(HD)
        if sparse[L]:
            t[a + "index_q_proj.weight"], t[a + "index_k_proj.weight"] = f(KV * HD, HID), f(HD, HID)
            t[a + "index_q_norm.weight"], t[a + "index_k_norm.weight"] = f(HD), f(HD)
        if not moe[L]:
            for k, o, i in (("gate_proj", DENSE, HID), ("up_proj", DENSE, HID), ("down_proj", HID, DENSE)):
                t[p + "mlp." + k + ".weight"] = f(o, i)
            continue
        m = p + "block_sparse_moe."
        t[m + "gate.weight"], t[m + "e_score_correction_bias"] = f(EXPERTS, HID), f(EXPERTS)
        for k, o, i in (("gate_proj", SHARED, HID), ("up_proj", SHARED, HID), ("down_proj", HID, SHARED)):
            t[m + "shared_experts." + k + ".weight"] = f(o, i)
        for k, o, i in (("gate_proj", INTER, HID), ("up_proj", INTER, HID), ("down_proj", HID, INTER)):
            t[m + "switch_mlp." + k + ".weight"] = f(EXPERTS, o, i)
    return t


def load_model(mm, mx, config, weights):
    m = mm.Model(mm.ModelArgs.from_dict(config))
    w = {k: mx.array(v) if isinstance(v, np.ndarray) else v for k, v in weights.items()}
    m.load_weights(list(m.sanitize(w).items()), strict=True)
    mx.eval(m.parameters())
    return m


def run_paths(m, mx, tokens, prefix, chunk):
    """Logits for full prefill (no cache), decode through the cache after `prefix`, and chunked prefill."""
    toks = mx.array(tokens)[None]
    full = np.array(m(toks).astype(mx.float32))[0]
    cache = m.make_cache()
    m(toks[:, :prefix], cache=cache)
    decode = np.concatenate([np.array(m(toks[:, i:i + 1], cache=cache).astype(mx.float32))[0]
                             for i in range(prefix, len(tokens))])
    cache = m.make_cache()
    chunked = np.concatenate([np.array(m(toks[:, i:i + chunk], cache=cache).astype(mx.float32))[0]
                              for i in range(0, len(tokens), chunk)])
    return full, decode, chunked, cache


def build_float_model(mm, cache_mod, mx, out_dir: Path, name, sparse, moe, first_seed, reference, report):
    """The first seed whose top-k choices all sit at least MIN_MARGIN from a tie, so that rounding in a strict float32
    consumer cannot flip a selected block or expert and turn a precision difference into a different model."""
    config = vl_config(text_config(LAYERS, sparse, moe, **FLOAT_SHAPES))
    for seed in range(first_seed, first_seed + 1000):
        weights = float_checkpoint(seed, sparse, moe)
        m = load_model(mm, mx, config, weights)
        tokens = np.random.default_rng(seed + 100_000).integers(0, VOCAB, SEQ).astype(np.int32)
        with Recorder(mm) as rec:
            m(mx.array(tokens)[None])
        checked = check_selection(rec, SEQ, name, MIN_MARGIN)
        expert_margin = min(router_margin(mod, x) for mod, x in rec.router)
        if checked is not None and expert_margin >= MIN_MARGIN:
            masks, disagree, block_margins = checked
            break
    else:
        raise AssertionError(f"{name}: no seed with top-k margins >= {MIN_MARGIN}")

    full, decode, chunked, cache = run_paths(m, mx, tokens, PREFIX, CHUNK)
    scale = float(np.abs(full).max())
    decode_err = float(np.abs(decode - full[PREFIX:]).max() / scale)
    chunk_err = float(np.abs(chunked - full).max() / scale)
    assert decode_err < 1e-5 and chunk_err < 1e-5, f"{name}: cache paths drift from prefill"
    layout = ["sparse" if isinstance(c, cache_mod.MiniMaxM3SparseCache) else "kv" for c in cache]
    assert layout == ["sparse" if s else "kv" for s in sparse], f"{name}: cache layout {layout}"
    assert np.isfinite(full).all()

    d = out_dir / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_checkpoint(d / "model.safetensors", {k: (v, "F32") for k, v in weights.items()})
    p = name + "."
    reference[p + "tokens"] = tokens
    reference[p + "logits"] = full
    reference[p + "logits_decode"] = decode
    reference[p + "logits_chunked"] = chunked
    reference[p + "key_masks"] = masks
    report[name] = {"seed": seed, "seq": SEQ, "prefix": PREFIX, "chunk": CHUNK, "sparse_attention_freq": sparse,
                    "moe_layer_freq": moe, "cache_layout": layout, "group_disagreements_per_sparse_layer": disagree,
                    "min_block_margin_per_sparse_layer": block_margins, "min_expert_margin": expert_margin,
                    "decode_vs_prefill_rel": decode_err, "chunked_vs_prefill_rel": chunk_err}


def source_checkpoint(seed):
    """(stored tensors {name: (array, dtype)}, float64 spec-dequant weights under the model's names)."""
    rng = np.random.default_rng(seed)
    f = lambda *s: rng.standard_normal(s).astype(np.float32) * 0.2  # noqa: E731
    bf16 = lambda v: v.astype(np.float32)  # noqa: E731 - rounded on write; the reference re-reads the stored values
    src, flt = {}, {}
    for name, v in (("model.embed_tokens.weight", f(S_VOCAB, S_HID)), ("lm_head.weight", f(S_VOCAB, S_HID)),
                    ("model.norm.weight", f(S_HID))):
        src["language_model." + name] = (bf16(v), "BF16")
        flt[name] = ("BF16", "language_model." + name)

    def lin(src_name, model_name, out_d, in_d):
        w, s, ref = mxfp8(rng, out_d, in_d)
        src[src_name + ".weight"], src[src_name + ".weight_scale_inv"] = (w, "F8_E4M3"), (s, "F8_E8M0")
        flt[model_name + ".weight"] = ref

    for L in range(2):
        p, q = f"language_model.model.layers.{L}.", f"model.layers.{L}."
        for k in ("input_layernorm", "post_attention_layernorm"):
            src[p + k + ".weight"] = (f(S_HID), "BF16")
            flt[q + k + ".weight"] = ("BF16", p + k + ".weight")
        a = "self_attn."
        for k, o, i in (("q_proj", S_HEADS * S_HD, S_HID), ("k_proj", S_KV * S_HD, S_HID),
                        ("v_proj", S_KV * S_HD, S_HID), ("o_proj", S_HID, S_HEADS * S_HD)):
            lin(p + a + k, q + a + k, o, i)
        lin(p + a + "index_q_proj", q + a + "indexer.index_q_proj", S_KV * S_HD, S_HID)
        lin(p + a + "index_k_proj", q + a + "indexer.index_k_proj", S_HD, S_HID)
        for k, mk in (("q_norm", "q_norm"), ("k_norm", "k_norm"), ("index_q_norm", "indexer.index_q_norm"),
                      ("index_k_norm", "indexer.index_k_norm")):
            src[p + a + k + ".weight"] = (f(S_HD), "BF16")
            flt[q + a + mk + ".weight"] = ("BF16", p + a + k + ".weight")
        if L == 0:
            for k, o, i in (("gate_proj", S_HID, S_HID), ("up_proj", S_HID, S_HID), ("down_proj", S_HID, S_HID)):
                lin(p + "mlp." + k, q + "mlp." + k, o, i)
            continue
        m = p + "block_sparse_moe."
        src[m + "gate.weight"] = (f(S_E, S_HID), "F32")
        src[m + "e_score_correction_bias"] = (f(S_E), "F32")
        flt[q + "mlp.gate.weight"] = ("F32", m + "gate.weight")
        flt[q + "mlp.e_score_correction_bias"] = ("F32", m + "e_score_correction_bias")
        for k, o, i in (("gate_proj", S_INTER, S_HID), ("up_proj", S_INTER, S_HID), ("down_proj", S_HID, S_INTER)):
            lin(m + "shared_experts." + k, q + "mlp.shared_experts." + k, o, i)
        for wn, proj, o, i in (("w1", "gate_proj", S_INTER, S_HID), ("w3", "up_proj", S_INTER, S_HID),
                               ("w2", "down_proj", S_HID, S_INTER)):
            stack = []
            for e in range(S_E):
                packed, scale, g, ref = nvfp4(rng, o, i)
                b = f"{m}experts.{e}.{wn}."
                src[b + "weight_packed"], src[b + "weight_scale"] = (packed, "U8"), (scale, "F8_E4M3")
                src[b + "weight_global_scale"] = (np.array([g], np.float32), "F32")
                src[b + "input_global_scale"] = (np.array([1.0], np.float32), "F32")
                stack.append(ref)
            flt[q + f"mlp.switch_mlp.{proj}.weight"] = np.stack(stack)
    # tensors a text-only load must drop
    src["vision_tower.patch_embedding.weight"] = (f(8, 3, 2, 2, 2), "BF16")
    src["multi_modal_projector.linear_1.weight"] = (f(S_HID, 8), "BF16")
    src["patch_merge_mlp.0.weight"] = (f(8, 8), "BF16")
    src["mtp.0.norm.weight"] = (f(S_HID), "BF16")
    return src, flt


def build_source_model(mm, sf, mx, out_dir: Path, reference, report, seed=0):
    src, flt_spec = source_checkpoint(seed)
    config = vl_config(text_config(2, [1, 1], [0, 1], **SOURCE_SHAPES), quantization_config={
        "quant_method": "mxfp8", "weight_block_size": [1, 32], "activation_scheme": "dynamic",
        "moe_quant_algo": "NVFP4", "moe_quant_format": "nvfp4-pack-quantized", "moe_group_size": 16,
        "ignored_layers": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector",
                           "patch_merge_mlp", "block_sparse_moe.gate"]})
    d = out_dir / "m31-source"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shard = d / "model.safetensors"
    save_checkpoint(shard, src)
    dtypes = safetensors_dtypes(shard)
    assert {"F8_E4M3", "F8_E8M0", "U8", "BF16", "F32"} <= set(dtypes.values()), dtypes

    stored = mx.load(str(shard))  # exactly what a runtime reads back (BF16 rounding included)
    assert sf.is_source_checkpoint(stored)
    # float spec dequant: BF16/F32 tensors as stored, quantized ones from the format definitions
    flt = {k: (np.array(stored[v[1]].astype(mx.float32)) if isinstance(v, tuple) else v.astype(np.float32))
           for k, v in flt_spec.items()}

    tokens = np.random.default_rng(3).integers(0, S_VOCAB, S_SEQ).astype(np.int32)
    as_stored = load_model(mm, mx, config, stored)
    sw = as_stored.model.layers[1].mlp.switch_mlp
    assert all(isinstance(getattr(sw, n), sf.NVFP4SwitchLinear) for n in ("gate_proj", "up_proj", "down_proj"))
    assert as_stored.model.layers[0].self_attn.q_proj.mode == "mxfp8"
    g = [np.array(getattr(sw, n).inv_global_scale) for n in ("gate_proj", "up_proj", "down_proj")]
    assert all(len(np.unique(v)) == S_E for v in g), "per-expert global scales must differ"

    with Recorder(mm) as rec:
        bf16_logits = np.array(as_stored(mx.array(tokens)[None]).astype(mx.float32))[0]
    _, _, bf16_margins = check_selection(rec, S_SEQ, "m31-source (bf16)")
    toks = mx.array(tokens)[None]
    cache = as_stored.make_cache()
    as_stored(toks[:, :S_PREFIX], cache=cache)
    bf16_decode = np.concatenate([np.array(as_stored(toks[:, i:i + 1], cache=cache).astype(mx.float32))[0]
                                  for i in range(S_PREFIX, S_SEQ)])

    # the same stored bytes with the BF16 tensors widened to F32, against the float spec-dequant model
    widened = {k: v.astype(mx.float32) if v.dtype == mx.bfloat16 else v for k, v in stored.items()}
    src32 = load_model(mm, mx, config, widened)
    with Recorder(mm) as rec:
        f32_logits = np.array(src32(toks).astype(mx.float32))[0]
    _, f32_disagree, f32_margins = check_selection(rec, S_SEQ, "m31-source (f32)")
    expert_margin = min(router_margin(mod, x) for mod, x in rec.router)
    assert min(f32_margins + bf16_margins) >= MIN_MARGIN and expert_margin >= MIN_MARGIN, "source: top-k near a tie"
    ref_model = load_model(mm, mx, config, {"language_model." + k: v for k, v in flt.items()})
    ref_logits = np.array(ref_model(toks).astype(mx.float32))[0]
    scale = float(np.abs(ref_logits).max())
    spec_err = float(np.abs(f32_logits - ref_logits).max() / scale)
    assert spec_err < 1e-5, f"source model vs float spec dequant {spec_err:.2e}"
    bf16_err = float(np.abs(bf16_logits - ref_logits).max() / scale)
    for n in ("gate_proj", "up_proj", "down_proj"):
        p = getattr(src32.model.layers[1].mlp.switch_mlp, n)
        p.inv_global_scale = p.inv_global_scale * 0
    no_experts = float(np.abs(np.array(src32(toks))[0] - ref_logits).max() / scale)
    assert no_experts > 100 * max(spec_err, 1e-6), "routed experts barely change the output"

    p = "m31-source."
    reference[p + "tokens"] = tokens
    reference[p + "logits_bf16"] = bf16_logits
    reference[p + "logits_bf16_decode"] = bf16_decode
    reference[p + "logits_f32"] = f32_logits
    reference[p + "logits_spec_dequant"] = ref_logits
    report["m31-source"] = {
        "seq": S_SEQ, "prefix": S_PREFIX, "stored_dtypes": sorted(set(dtypes.values())),
        "f32_vs_spec_dequant_rel": spec_err, "bf16_vs_spec_dequant_rel": bf16_err,
        "experts_zeroed_vs_spec_dequant_rel": no_experts,
        "group_disagreements_per_sparse_layer": f32_disagree,
        "min_block_margin_per_sparse_layer_f32": f32_margins,
        "min_block_margin_per_sparse_layer_bf16": bf16_margins, "min_expert_margin_f32": expert_margin,
        "inv_global_scales": {n: v.tolist() for n, v in zip(("gate_proj", "up_proj", "down_proj"), g)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vmlx", type=Path, required=True, help="vMLX checkout providing vmlx_engine")
    parser.add_argument("--vmlx-commit", required=True, help="commit of that checkout, recorded for provenance")
    parser.add_argument("--output", type=Path, required=True, help="new, empty output directory")
    args = parser.parse_args()
    out = args.output.resolve()
    fixtures = Path(__file__).resolve().parents[2] / "Tests/MLXLMTests/Fixtures/MiniMaxM3"
    if out == fixtures.resolve():
        parser.error("generate into a separate directory and compare before replacing the fixtures")
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.vmlx.resolve()))

    import mlx.core as mx
    import mlx_lm
    from vmlx_engine.models.minimax_m3 import cache as cache_mod
    from vmlx_engine.models.minimax_m3 import minimax_m3 as mm
    from vmlx_engine.models.minimax_m3 import source_format as sf

    a = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    tf32_probe = float(np.abs(np.array(mx.array(a) @ mx.array(a)) - a.astype(np.float64) @ a).max())
    assert tf32_probe < 1e-3, f"float32 matmul error {tf32_probe:.1e}: MLX_ENABLE_TF32=0 did not take effect"

    reference: dict = {}
    report: dict = {}
    build_attention(mm, mx, out, reference, report)
    build_float_model(mm, cache_mod, mx, out, "m31-float", [1] * LAYERS, [0, 0, 0, 1, 1], 10, reference, report)
    build_float_model(mm, cache_mod, mx, out, "m3-float", [0, 0, 0, 1, 1], [0, 0, 0, 1, 1], 20, reference, report)
    build_source_model(mm, sf, mx, out, reference, report)

    mx.save_safetensors(str(out / "reference.safetensors"), {k: mx.array(v) for k, v in reference.items()})
    src_dir = args.vmlx.resolve() / "vmlx_engine/models/minimax_m3"
    provenance = {
        "generator": "scripts/minimax_m3_fixtures/generate_fixtures.py",
        "generator_sha256": sha256(Path(__file__).resolve()),
        "vmlx_commit": args.vmlx_commit,
        "reference_sha256": {name: sha256(src_dir / name) for name in SOURCES},
        "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "machine": platform.machine(),
        "float32_matmul": f"MLX_ENABLE_TF32=0 (64x64 probe max abs error {tf32_probe:.1e})",
        "checks": report,
    }
    (out / "reference.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
