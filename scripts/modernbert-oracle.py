#!/usr/bin/env python3
r"""Reference fixtures for the ModernBERT port.

Both subcommands run the Hugging Face reference (transformers' ModernBertModel with eager attention, on
the CPU) and write what the port's tests compare against. Development tooling: nothing here is part of
any build.

tier1 builds a tiny ModernBERT with unit-gain weights. It alone regenerates the fixture of vmlx's
Tests/MLXLMTests/ModernBertTests.swift and ModernBertQuantizationTests.swift, the two files
Tests/MLXLMTests/Resources/modernbert-tiny.{safetensors,json}, offline and in seconds:
  - modernbert-tiny.safetensors holds the model's parameters under Hugging Face's names, and float32
    references: for each of the cases standard, aligned and short, ref.<case>.input_ids and
    .attention_mask (int32, batch x length), .layer0, .layer1 and .layer3 (those layers' outputs, the
    last taken before final_norm), .final (the last hidden state) and .pooled (its position 0); the
    same input_ids, final and pooled for the clamp case; and ref.mlp.input and ref.mlp.output for one
    layer's MLP alone.
  - modernbert-tiny.json holds the recorded conditions, the configuration and the clamp case's, each
    case's lengths, the bounds, and each mutation's measured effect on the reference.

tier2 runs the real Granite Embedding R2 checkpoints, unquantized and 8-bit, each in float32 and in
bfloat16, and writes reference vectors for downstream checkpoint tests:
  - granite-r2-reference.safetensors holds base.fp32, base.bf16, q8.fp32 and q8.bf16, each (11, 768)
    float32: one CLS vector per text (the last hidden state at position 0), unnormalized. It also holds
    base.layers.bf16 and q8.layers.bf16, each (23, 4, 768) float32: the bfloat16 models' states for
    text 0 at its first TRACE_POSITIONS positions, the embedding output and then the 22 layers'
    outputs, the last taken before final_norm.
  - granite-r2-reference.json holds the recorded conditions, both checkpoints and their revisions, the
    texts, the batches they ran in, their token ids and TRACE_POSITIONS.

Both outputs are byte-identical only under the conditions their JSON records. Elsewhere they may differ
by rounding: at float32's level in the float32 references, at bfloat16's in the bfloat16 ones.

Setup, in a throwaway virtual environment on Python 3.12:
    python3.12 -m venv <venv>
    <venv>/bin/pip install "torch==2.14.0" "transformers==4.57.3" "tokenizers==0.22.2" \
        safetensors numpy huggingface_hub

tier2's inputs, at the revisions pinned below (tier2 checks their hashes before loading anything):
    <venv>/bin/hf download ibm-granite/granite-embedding-311m-multilingual-r2 \
        --revision 44399559930365213510b1ee2eb15ded83374f0e --local-dir <base>
    <venv>/bin/hf download beaupi/granite-embedding-311m-multilingual-r2-oQ8 \
        --revision e0b3c484d2897406c438ba888a948874aaefc940 --local-dir <q8>

Run, with the environment's own interpreter:
    <venv>/bin/python <this script> tier1 --out <dir>
    <venv>/bin/python <this script> tier2 --base <base> --q8 <q8> --out <dir>
tier2 takes about 25 minutes on an idle M4 Max, nearly all of it in the two bfloat16 models' passes over
the 8192-token text: torch's CPU bfloat16 batched matmul, which eager attention uses, has no fast path on
macOS.
"""

import argparse
import copy
import hashlib
import json
import math
import platform
import struct
from pathlib import Path

# Read before the slow imports below, so an edit made while they load cannot reach the hash;
# conditions() hashes these bytes rather than re-reading the file.
GENERATOR_BYTES = Path(__file__).read_bytes()

import numpy as np
import tokenizers
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel


def check(ok, message):
    """Validation that `python -O` cannot strip, as it strips assert."""
    if not ok:
        raise AssertionError(message)


SEED = 20260918
# The largest absolute differences, in float32, that vmlx's ModernBertTests.swift allows between the port
# and this reference: MODEL_BOUND for hidden states at unpadded positions and for pooled vectors,
# MLP_BOUND for one layer's MLP alone. The tests read both from the fixture JSON, so these lines are
# their only definition. The MLP case exists because exact and tanh GELU differ by at most 4.7e-4, which
# the whole-model bound cannot see; in the MLP case the tanh form moves the output by about 1.5e-3,
# fifteen times MLP_BOUND, against float32 noise near 1e-5.
MODEL_BOUND = 2e-4
MLP_BOUND = 1e-4
MIN_RATIO = 10.0  # a numerical mutation must clear ten times its bound

# A local layer attends within local_attention // 2 positions, so in a batch padded to max(lengths) the
# shortest item's last padded row has no real key in reach once min <= max - local_attention // 2 - 1
# (check_layout). That fully masked row is the case that matters: with an additive mask, MLX's fused
# attention kernels compute 0/0 for it when the key length fills whole KEY_TILE-key tiles, and the NaN
# reaches the item's real rows through the P·V product. Boolean masks cannot cause this. The row's value
# then depends on which kernel ran, so padded rows are checked for finiteness but never compared.
KEY_TILE = 32  # keys per tile of MLX's fused attention kernels at head dimension 64
STANDARD_LENGTHS = [16, 11]  # the short item as long as the window allows
# The same layout at exactly one full tile of keys: the path on which Blaizzy/mlx-embeddings#80 turns
# padded items into NaN.
ALIGNED_LENGTHS = [32, 11]
# At most this many query tokens reach MLX's short-query attention kernel, the path a single search query
# takes: MLX 0.32.2's scaled_dot_product_attention.cpp switches to the full kernel above 8 query tokens,
# choosing by the padded width.
SHORT_QUERY_MAX_TOKENS = 8
SHORT_LENGTHS = [6, 3]
CLAMP_INPUT = 20
CLAMP_LIMIT = 12
MLP_LAYER = 1
MLP_TOKENS = 64
MLP_RANGE = (-3.0, 3.0)

TINY = dict(
    vocab_size=64, hidden_size=128, intermediate_size=192, num_hidden_layers=4,
    num_attention_heads=2, max_position_embeddings=64, local_attention=8,
    global_attn_every_n_layers=3, global_rope_theta=150000.0, local_rope_theta=160000.0,
    norm_eps=1e-12, hidden_activation="gelu", pad_token_id=0, bos_token_id=2, eos_token_id=1,
    cls_token_id=2, sep_token_id=1, reference_compile=False,
)


def check_layout(name, lengths, local_attention, tiled=False):
    """A padded batch layout must produce the fully masked row it exists for (see KEY_TILE)."""
    reach = local_attention // 2
    check(min(lengths) <= max(lengths) - reach - 1,
          f"{name} {lengths}: the short item's last padded row has a real key within {reach} positions, "
          f"so no local-layer row is fully masked and the layout loses the case it exists for")
    if tiled:
        check(max(lengths) % KEY_TILE == 0,
              f"{name} {lengths}: {max(lengths)} keys do not fill whole {KEY_TILE}-key tiles, the only "
              f"layout on which additive masks turn padded items into NaN")


check_layout("STANDARD_LENGTHS", STANDARD_LENGTHS, TINY["local_attention"])
check_layout("ALIGNED_LENGTHS", ALIGNED_LENGTHS, TINY["local_attention"], tiled=True)
check(max(SHORT_LENGTHS) <= SHORT_QUERY_MAX_TOKENS,
      f"SHORT_LENGTHS {SHORT_LENGTHS}: over {SHORT_QUERY_MAX_TOKENS} tokens, the short case would take "
      f"MLX's full attention kernel and never reach the short-query one")


def git_blob_id(data):
    """The id git, and the Hugging Face Hub for files outside LFS, give these bytes."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def conditions():
    """What the reference's numbers depend on, recorded with every fixture.

    The checkpoint tests' bounds are multiples of the reference's own bfloat16 drift, and that drift
    depends on how the reference runs: the libraries, the platform (and with it torch's BLAS), the device
    and the attention implementation. The generator's hashes find the version that wrote a fixture
    through `git log --find-object`.
    """
    return {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "device": "cpu", "attn_implementation": "eager",
        "generator_sha256": hashlib.sha256(GENERATOR_BYTES).hexdigest(),
        "generator_git_blob": git_blob_id(GENERATOR_BYTES),
    }


def build(config):
    model = ModernBertModel._from_config(config, attn_implementation="eager")
    check(model.config._attn_implementation == "eager",
          f"built with {model.config._attn_implementation} attention, not the eager attention "
          f"conditions() records")
    # Always float32: _from_config builds under the config's own dtype, which is bfloat16 for
    # Granite R2, and tier 2's float32 8-bit model is built here.
    return model.to(torch.float32).train(False)


def unit_gain(model, seed):
    """Fan-in-scaled weights and LayerNorm weights drawn away from 1.

    Hugging Face's own initialization (std 0.02, norms of exactly 1) keeps activations so small that
    three of the mutations become invisible.
    """
    g = torch.Generator().manual_seed(seed)
    hidden, inter = model.config.hidden_size, model.config.intermediate_size
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("tok_embeddings.weight"):
                p.copy_(torch.randn(p.shape, generator=g))
            elif name.endswith("norm.weight"):
                p.copy_(torch.rand(p.shape, generator=g) + 0.5)
            elif name.endswith("mlp.Wo.weight"):
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(inter))
            elif name.endswith(("attn.Wqkv.weight", "attn.Wo.weight", "mlp.Wi.weight")):
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(hidden))
            else:
                raise ValueError(f"unit_gain: no rule for {name}")


def run(model, ids, mask):
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    return out.last_hidden_state, out.hidden_states


def batch(lengths, total, vocab, g):
    ids = torch.zeros((len(lengths), total), dtype=torch.long)
    mask = torch.zeros((len(lengths), total), dtype=torch.long)
    for row, n in enumerate(lengths):
        ids[row, :n] = torch.randint(3, vocab, (n,), generator=g)
        mask[row, :n] = 1
    return ids, mask


def model_effect(reference, mutated, ids, mask):
    """Largest change at unpadded positions: what tier 1 compares.

    Position 0 is always a real token, because padding is on the right, so it is already covered by
    the unpadded-position comparison above; a separate pooled-vector term would be redundant.
    """
    last, _ = run(mutated, ids, mask)
    real = mask.bool()
    return (last - reference).abs()[real].max().item()


def tier1(out_dir):
    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED + 1)
    config = ModernBertConfig(**TINY)
    model = build(config)
    unit_gain(model, SEED)
    hidden, inter = config.hidden_size, config.intermediate_size

    tensors = {name: p.detach().clone() for name, p in model.named_parameters()}

    def record(case, ids, mask):
        last, states = run(model, ids, mask)
        check(torch.isfinite(last).all(),
              f"{case}: the reference produced non-finite values, so the case holds the port to nothing")
        tensors[f"ref.{case}.input_ids"] = ids.to(torch.int32).contiguous()
        tensors[f"ref.{case}.attention_mask"] = mask.to(torch.int32).contiguous()
        tensors[f"ref.{case}.layer0"] = states[1].contiguous()
        tensors[f"ref.{case}.layer1"] = states[2].contiguous()
        # The last layer's output before final_norm: separates a final_norm fault from a layer fault.
        tensors[f"ref.{case}.layer3"] = states[4].contiguous()
        tensors[f"ref.{case}.final"] = last.contiguous()
        # clone(), not contiguous(): a slice that is already contiguous shares memory with `final`,
        # and safetensors refuses to save tensors that share memory.
        tensors[f"ref.{case}.pooled"] = last[:, 0].clone()
        return last

    standard_ids, standard_mask = batch(STANDARD_LENGTHS, max(STANDARD_LENGTHS), config.vocab_size, g)
    standard = record("standard", standard_ids, standard_mask)
    aligned_ids, aligned_mask = batch(ALIGNED_LENGTHS, max(ALIGNED_LENGTHS), config.vocab_size, g)
    record("aligned", aligned_ids, aligned_mask)

    clamp_ids, _ = batch([CLAMP_INPUT], CLAMP_INPUT, config.vocab_size, g)
    # Drawn after every other case's ids, so no existing tensor changes.
    short_ids, short_mask = batch(SHORT_LENGTHS, max(SHORT_LENGTHS), config.vocab_size, g)
    record("short", short_ids, short_mask)
    check(clamp_ids.shape[1] > CLAMP_LIMIT,
          f"CLAMP_INPUT {CLAMP_INPUT} does not exceed CLAMP_LIMIT {CLAMP_LIMIT}, so the clamp case cannot "
          f"tell the configured clamp from a literal one")
    tensors["ref.clamp.input_ids"] = clamp_ids.to(torch.int32).contiguous()
    clamped, _ = run(model, clamp_ids[:, :CLAMP_LIMIT], torch.ones((1, CLAMP_LIMIT), dtype=torch.long))
    tensors["ref.clamp.final"] = clamped.contiguous()
    tensors["ref.clamp.pooled"] = clamped[:, 0].clone()  # a batch of one: see record()

    mlp_in = torch.linspace(*MLP_RANGE, MLP_TOKENS * hidden).reshape(1, MLP_TOKENS, hidden)
    with torch.no_grad():
        mlp_out = model.layers[MLP_LAYER].mlp(mlp_in)
    tensors["ref.mlp.input"] = mlp_in.contiguous()
    tensors["ref.mlp.output"] = mlp_out.contiguous()

    # The port's mutations that tier 1 must each catch, numbered as the JSON's `mutations` and
    # `outright_failures` keys are:
    #   1. swap the local and global RoPE θ;
    #   2. make every layer global;
    #   3. give layer 0 a real attn_norm, which fails at weight binding: no checkpoint has a
    #      layers.0.attn_norm.weight to bind;
    #   4. clamp the input length to a literal 32768 rather than the configuration's max_position_embeddings,
    #      which the clamp case rejects on shape alone;
    #   5. use tanh GELU (MLX's `.precise`) rather than exact GELU, caught by the MLP case;
    #   6. swap the two halves of Wi's output;
    #   7. build the masks additively, filled with the dtype's most negative finite value, which must
    #      turn the aligned case's padded items into NaN.
    # 1 to 6 are measured on the reference below, each numerical one against MIN_RATIO times its bound,
    # and 3 is also shown to fail outright. 7 is a property of MLX's kernels, with no reference
    # counterpart.
    effects = {}

    swapped = build(ModernBertConfig(**{**TINY, "global_rope_theta": TINY["local_rope_theta"],
                                        "local_rope_theta": TINY["global_rope_theta"]}))
    swapped.load_state_dict(model.state_dict())
    effects["1_swap_theta"] = (model_effect(standard, swapped, standard_ids, standard_mask), MODEL_BOUND)

    all_global = build(ModernBertConfig(**{**TINY, "global_attn_every_n_layers": 1}))
    all_global.load_state_dict(model.state_dict())
    effects["2_every_layer_global"] = (
        model_effect(standard, all_global, standard_ids, standard_mask), MODEL_BOUND)

    normed = copy.deepcopy(model)
    normed.layers[0].attn_norm = torch.nn.LayerNorm(hidden, eps=config.norm_eps, bias=False)
    effects["3_layer0_attn_norm"] = (
        model_effect(standard, normed, standard_ids, standard_mask), MODEL_BOUND)

    # Mutation 3 must fail outright, and on the reference too: a real layer-0 attn_norm has no weight
    # in the checkpoint, so binding refuses.
    outright = {}
    try:
        normed.load_state_dict(model.state_dict())
    except RuntimeError as e:
        check("layers.0.attn_norm.weight" in str(e),
              f"mutation 3 failed to bind, but not over layers.0.attn_norm.weight: {e}")
        outright["3_layer0_attn_norm"] = "load_state_dict refuses: no layers.0.attn_norm.weight"
    else:
        raise AssertionError("mutation 3: the weights bound to a real layer-0 attn_norm")

    unclamped, _ = run(model, clamp_ids, torch.ones((1, CLAMP_INPUT), dtype=torch.long))
    effects["4_clamp_literal"] = ((unclamped[:, 0] - clamped[:, 0]).abs().max().item(), MODEL_BOUND)

    tanh_mlp = copy.deepcopy(model.layers[MLP_LAYER].mlp)
    tanh_mlp.act = torch.nn.GELU(approximate="tanh")
    with torch.no_grad():
        effects["5_gelu_tanh_mlp_case"] = ((tanh_mlp(mlp_in) - mlp_out).abs().max().item(), MLP_BOUND)

    halves = copy.deepcopy(model)
    with torch.no_grad():
        for layer in halves.layers:
            w = layer.mlp.Wi.weight.detach().clone()
            layer.mlp.Wi.weight.copy_(torch.cat([w[inter:], w[:inter]]))
    effects["6_swap_wi_halves"] = (model_effect(standard, halves, standard_ids, standard_mask), MODEL_BOUND)

    report = {}
    for name, (effect, bound) in effects.items():
        ratio = effect / bound
        report[name] = {"effect": effect, "bound": bound, "ratio": ratio}
        print(f"  mutation {name}: effect {effect:.3e}, {ratio:.1f}x its bound")
        # Mutation 3 fails at weight binding in Swift, so its reference effect is informative only.
        if not name.startswith("3_"):
            check(ratio >= MIN_RATIO,
                  f"mutation {name} clears its bound by only {ratio:.1f}x, under {MIN_RATIO:.0f}x, so "
                  f"float32 noise could hide it from the port's tests")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tensors.items():
        if t.is_floating_point():
            check(torch.isfinite(t).all(), f"{name}: contains a non-finite value, which no test can compare")
    save_file(tensors, str(out_dir / "modernbert-tiny.safetensors"))
    config_json = {**TINY, "model_type": "modernbert"}
    meta = {
        "conditions": {**conditions(), "seed": SEED},
        "config": config_json,
        "clamp_config": {**config_json, "max_position_embeddings": CLAMP_LIMIT},
        "cases": {"standard": {"lengths": STANDARD_LENGTHS}, "aligned": {"lengths": ALIGNED_LENGTHS},
                  "clamp": {"input_length": CLAMP_INPUT, "limit": CLAMP_LIMIT},
                  "short": {"lengths": SHORT_LENGTHS}},
        "mlp_case": {"layer": MLP_LAYER, "tokens": MLP_TOKENS, "range": MLP_RANGE},
        "bounds": {"model": MODEL_BOUND, "mlp": MLP_BOUND},
        "mutations": report,
        "outright_failures": outright,
    }
    (out_dir / "modernbert-tiny.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"tier 1 fixture written to {out_dir}")


BASE_MODEL = "ibm-granite/granite-embedding-311m-multilingual-r2"
BASE_REVISION = "44399559930365213510b1ee2eb15ded83374f0e"
Q8_MODEL = "beaupi/granite-embedding-311m-multilingual-r2-oQ8"
Q8_REVISION = "e0b3c484d2897406c438ba888a948874aaefc940"
# What those revisions hold, as the Hub identifies each file: sha256 for files in LFS, the git blob id for
# the rest. They cover every file tier2 reads, so a download from any other revision fails before anything
# is loaded instead of producing references that claim a revision they were not computed from.
BASE_FILES = {
    "config.json": ("git_blob", "fe4dfa0116f63917ad77d6a82fd9cd139903cf0d"),
    "model.safetensors": ("sha256", "dcb6431bfa6e817fe100a2b0521360cec3383963b03fa966b685de18ca310d31"),
    "special_tokens_map.json": ("git_blob", "3338497239b49ce5372b4e70019dd63a88ef227a"),
    "tokenizer.json": ("sha256", "0087c868b33bad550a78a08d19798cfd7f713cde4f020803b8f51f405503e15f"),
    "tokenizer_config.json": ("git_blob", "b71d75c5c640f4ca16679395b8e210d2514dac7e"),
}
Q8_FILES = {
    "config.json": ("git_blob", "8b337a9f5fa1ac0b9a8f7306356571bbbc4d2efe"),
    "model.safetensors": ("sha256", "e795a2a423868214b1a2a3d2a7773341094bcd92d43eab829db0db26ee0ccbe7"),
}
# Where the 8-bit conversion's config.json may differ from the base's. The 8-bit reference runs under the
# base's configuration, which is sound because none of these reaches ModernBertModel's arithmetic:
# classifier_pooling configures heads it does not have, the two quantization blocks describe the packed
# weights dequantized_state() unpacks, and the reference computes RoPE for any position, while the port
# clamps at max_position_embeddings, which is why the long text stays within both checkpoints' values.
Q8_CONFIG_DIFFERENCES = (
    "classifier_pooling", "max_position_embeddings", "quantization", "quantization_config")

# Eight short texts: seven sentences of prose, in English, French, Arabic, Chinese, Japanese, Hindi and
# Russian, which also make up the long text, and one source-code snippet.
PROSE_TEXTS = [
    "The committee postponed its decision until the survey results were published in spring.",
    "Le musée rouvrira ses portes au public après deux années de rénovation complète.",
    "أعلنت الجامعة عن برنامج جديد لدراسة تغير المناخ في المناطق الساحلية.",
    "研究人员在高原湖泊中发现了一种能够适应极端低温的新型微生物。",
    "図書館は来月から開館時間を延長し、夜間の学習スペースを提供します。",
    "किसानों ने इस वर्ष मानसून की देरी के कारण बुवाई का समय बदल दिया।",
    "Новая линия метро соединит аэропорт с центром города к концу года.",
]
TEXTS = PROSE_TEXTS + [
    "func clamp(_ x: Int, to range: ClosedRange<Int>) -> Int {\n"
    "    min(max(x, range.lowerBound), range.upperBound)\n}",
]
# The long text must cross the local window many times over, yet stay within both checkpoints'
# max_position_embeddings (the 8-bit conversion rewrites its own), so neither one's clamp engages.
LONG_TEXT_MIN_TOKENS = 8000
# Two more texts, run as one batch at exact token counts: the longer fills whole KEY_TILE-key tiles, the
# aligned kernel path, under the checkpoints' own local window (check_layout).
CHECKPOINT_ALIGNED_LENGTHS = [256, 150]
CHECKPOINT_ALIGNED_WORDS = ["granite ", "basalt "]
# Per-layer states kept for text 0. They localize a failed bound layer by layer: a failed bound is to be
# localized, never loosened.
TRACE_POSITIONS = 4


def count(tok, text):
    return len(tok(text)["input_ids"])


def long_text(tok, low, high):
    """PROSE_TEXTS repeated and cut to between `low` and `high` tokens, special tokens included.

    Bisects on the prefix length for a prefix at or under `high`. The token count is not monotone in the
    prefix, because the normalizer turns spaces into ▁ before splitting and BPE merges across words, so
    the bisection need not find the longest such prefix; the final check is what guarantees the range.
    """
    paragraph = " ".join(PROSE_TEXTS) + " "
    text = paragraph
    while count(tok, text) <= high:
        text += paragraph
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(tok, text[:mid]) <= high:
            lo = mid
        else:
            hi = mid - 1
    prefix = text[:lo]
    n = count(tok, prefix)
    check(low <= n <= high,
          f"the long text has {n} tokens, outside [{low}, {high}]: it must cross the local window many "
          f"times, yet engage neither checkpoint's position clamp")
    return prefix


def exact_tokens(tok, n, word):
    """`word` repeated and cut to exactly `n` tokens, special tokens included.

    Appends `word` until the text reaches `n` tokens, then drops characters until it is back at `n`. Each
    append must add a token, which bounds the first loop at `n` appends: a word that adds none, such as
    an empty one or a lone space, fails the check instead of looping. Dropping a character can split a
    token and skip past `n`, so the result is checked too.
    """
    text = ""
    for _ in range(n):
        if count(tok, text) >= n:
            break
        text += word
    check(count(tok, text) >= n,
          f"{n} appends of {word!r} reach only {count(tok, text)} tokens: each append must add one")
    while count(tok, text) > n and text:
        text = text[:-1]
    found = count(tok, text)
    check(found == n, f"{word!r} cut to {found} tokens, not {n}: the aligned batch needs exact lengths")
    return text


def padded(id_lists, pad_id):
    """Input ids and attention mask for one batch, padded on the right.

    Right padding keeps CLS at index 0 with positions counted from 0. The pad id itself does not matter,
    because padded keys are masked.
    """
    width = max(len(ids) for ids in id_lists)
    input_ids = torch.full((len(id_lists), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), width), dtype=torch.long)
    for row, ids in enumerate(id_lists):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[row, : len(ids)] = 1
    return input_ids, mask


def cls_vectors(model, token_ids, batches, pad_id):
    """Each text's unnormalized CLS vector, in float32, with the texts run in the given batches."""
    rows = [None] * len(token_ids)
    for group in batches:
        last, _ = run(model, *padded([token_ids[i] for i in group], pad_id))
        for row, i in enumerate(group):
            rows[i] = last[row, 0].float()
    return torch.stack(rows)


def layer_trace(model, ids, pad_id):
    """One text's states at its first TRACE_POSITIONS positions: the embedding output, then each layer's."""
    _, states = run(model, *padded([ids], pad_id))
    return torch.stack([s[0, :TRACE_POSITIONS].float() for s in states])


def check_rope_in_float32(model, name):
    """RoPE's frequencies must stay float32, as a native bfloat16 load leaves them.

    `inv_freq` is a non-persistent buffer, so casting a whole model with `.to(torch.bfloat16)` rounds it
    too. At 8k positions that rounding shifts RoPE phases by whole radians, inflating exactly the drift
    the checkpoint tests' bounds are built from. MLX computes RoPE frequencies in float32, so the port
    would not share the error.
    """
    freqs = [m.inv_freq for m in model.modules() if isinstance(getattr(m, "inv_freq", None), torch.Tensor)]
    check(freqs, f"{name}: no module holds an inv_freq tensor, so RoPE's precision goes unchecked; "
                 f"transformers has likely moved it, and this check must follow")
    dtypes = sorted({str(f.dtype) for f in freqs})
    check(dtypes == ["torch.float32"],
          f"{name}: RoPE frequencies in {dtypes}, not float32, which inflates the bfloat16 drift the "
          f"checkpoint tests' bounds are built from")


def check_files(directory, pinned, name):
    """Refuses a download that is not the pinned revision, before anything is loaded from it."""
    for filename, (kind, expected) in pinned.items():
        path = Path(directory) / filename
        if kind == "sha256":
            with path.open("rb") as f:
                found = hashlib.file_digest(f, "sha256").hexdigest()
        else:
            found = git_blob_id(path.read_bytes())
        check(found == expected,
              f"{name} {filename}: {kind} {found}, where the pinned revision has {expected}; the "
              f"references would claim a revision they were not computed from")


def check_configs(base_dir, q8_dir):
    """Returns the longest input both checkpoints accept unclamped, once their configs are equivalent."""
    base, q8 = (json.loads((Path(d) / "config.json").read_text(encoding="utf-8")) for d in (base_dir, q8_dir))
    differing = sorted(k for k in base.keys() | q8.keys() if base.get(k) != q8.get(k))
    unexpected = [k for k in differing if k not in Q8_CONFIG_DIFFERENCES]
    check(not unexpected,
          f"the 8-bit config.json differs from the base's in {unexpected}; its reference runs under the "
          f"base's configuration, which is sound only where the two agree")
    return min(base["max_position_embeddings"], q8["max_position_embeddings"])


def read_safetensors(path):
    """Raw reader: numpy cannot hold bfloat16, and the 8-bit weights are packed uint32."""
    data = Path(path).read_bytes()
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + n])
    header.pop("__metadata__", None)
    out = {}
    for name, info in header.items():
        begin, end = info["data_offsets"]
        raw = data[8 + n + begin : 8 + n + end]
        if info["dtype"] == "BF16":
            arr = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif info["dtype"] == "U32":
            arr = np.frombuffer(raw, dtype="<u4")
        elif info["dtype"] == "F32":
            arr = np.frombuffer(raw, dtype="<f4")
        else:
            raise ValueError(f"{name}: unexpected dtype {info['dtype']}")
        out[name] = arr.reshape(info["shape"])
    return out


def dequantized_state(q8_dir):
    """The published 8-bit weights as exact float32: affine is q * scale + bias per group."""
    quantization = json.loads((Path(q8_dir) / "config.json").read_text(encoding="utf-8"))["quantization"]
    check(quantization == {"group_size": 64, "bits": 8, "mode": "affine"},
          f"quantization {quantization}: the unpacking below reads 8-bit affine weights in groups of 64")
    group = quantization["group_size"]
    raw = read_safetensors(Path(q8_dir) / "model.safetensors")
    state = {}
    for name, arr in raw.items():
        if name.endswith((".scales", ".biases")):
            continue
        stem = name.removesuffix(".weight")
        if f"{stem}.scales" in raw:
            q = arr.view(np.uint8).astype(np.float32)  # four 8-bit values per uint32, little-endian
            rows, cols = q.shape
            scales, biases = raw[f"{stem}.scales"], raw[f"{stem}.biases"]
            check(cols % group == 0 and scales.shape == biases.shape == (rows, cols // group),
                  f"{stem}: scales {scales.shape} and biases {biases.shape} are not one pair per {group} "
                  f"of its {rows} x {cols} weights, so weights would be paired with another group's")
            w = q.reshape(rows, cols // group, group) * scales[..., None] + biases[..., None]
            state[name] = torch.from_numpy(w.reshape(rows, cols))  # w is new memory: no copy needed
        else:
            state[name] = torch.from_numpy(arr.astype(np.float32))  # the one copy, writable
    return state


def load_base(base_dir, dtype):
    """The unquantized checkpoint, refused unless every weight in it binds exactly."""
    model, info = ModernBertModel.from_pretrained(
        base_dir, attn_implementation="eager", reference_compile=False, dtype=dtype, output_loading_info=True)
    faults = {key: found for key, found in info.items() if found}
    check(not faults,
          f"loading the base checkpoint in {dtype}: {faults}; transformers fills a missing weight with "
          f"random values and only warns")
    check(model.config._attn_implementation == "eager",
          f"the base checkpoint in {dtype} runs {model.config._attn_implementation} attention, not the "
          f"eager attention conditions() records")
    return model.train(False)


def load_models(base_dir, q8_dir, config):
    """Both checkpoints, each in float32 and in bfloat16, keyed as the vectors they produce."""
    base32 = load_base(base_dir, torch.float32)
    # Loaded natively, as a bfloat16 user runs it: parameters in bfloat16, RoPE frequencies in float32.
    base16 = load_base(base_dir, torch.bfloat16)
    q32 = build(config)
    q32.load_state_dict(dequantized_state(q8_dir), strict=True)
    q16 = copy.deepcopy(q32)
    # q16's weights are bf16(q * scale + bias): the exact value, rounded once. On the GPU that is bit for bit
    # what MLX 0.32.2 uses wherever the port meets a quantized weight: the dequantize kernel behind the
    # embedding lookup and qmm both round once from float32, and qmv never rounds a weight. MLX's CPU backend
    # rounds twice in all three, bf16(bf16(q * scale) + bias), changing 65% of the weights; a q16 rounded
    # that way takes text 10 to 5.2 times its 1 - cosine drift, so on the CPU a correct port could fail.
    for p in q16.parameters():  # parameters only: a whole-model cast would round inv_freq too
        p.data = p.data.to(torch.bfloat16)
    models = {"base.fp32": base32, "base.bf16": base16, "q8.fp32": q32, "q8.bf16": q16}
    for key, model in models.items():
        dtype = torch.bfloat16 if key.endswith(".bf16") else torch.float32
        found = sorted({str(p.dtype) for p in model.parameters()})
        check(found == [str(dtype)],
              f"{key}: parameters in {found}, not {dtype}, so its vectors would not measure what their "
              f"key says")
        check_rope_in_float32(model, key)
    return models


def tier2_inputs(tok, max_positions):
    """The texts, their token ids and the batches they run in, as the JSON records them."""
    texts = TEXTS + [long_text(tok, LONG_TEXT_MIN_TOKENS, max_positions)]
    pairs = zip(CHECKPOINT_ALIGNED_LENGTHS, CHECKPOINT_ALIGNED_WORDS)
    texts += [exact_tokens(tok, n, word) for n, word in pairs]
    token_ids = [tok(t)["input_ids"] for t in texts]
    long_index = len(TEXTS)
    # The long text runs alone: padding the short texts to its length would cost about 29 GB of float32
    # attention weights per layer on the CPU.
    batches = [list(range(long_index)), [long_index], list(range(long_index + 1, len(texts)))]
    return texts, token_ids, batches


def tier2(base_dir, q8_dir, out_dir):
    check_files(base_dir, BASE_FILES, BASE_MODEL)
    check_files(q8_dir, Q8_FILES, Q8_MODEL)
    max_positions = check_configs(base_dir, q8_dir)
    config = ModernBertConfig.from_pretrained(base_dir, reference_compile=False)
    check_layout("CHECKPOINT_ALIGNED_LENGTHS", CHECKPOINT_ALIGNED_LENGTHS, config.local_attention, tiled=True)

    tok = AutoTokenizer.from_pretrained(base_dir)
    # Loading logs that this tokenizer has "an incorrect regex pattern" and should be loaded with
    # fix_mistral_regex=True. A false positive: transformers 4.57.3 says so of any local checkpoint saved
    # by a version above 4.57.2 and below 5.0, whatever its model type, and Granite's pre-tokenizer is a
    # plain Split on spaces with no regex to fix. The ids must be the tokenizer as shipped, the one vmlx
    # reads.
    check(not getattr(tok, "fix_mistral_regex", False),
          "the tokenizer was loaded with fix_mistral_regex, so its ids are not those of the shipped "
          "tokenizer, which vmlx reads")
    texts, token_ids, batches = tier2_inputs(tok, max_positions)
    no_cls = [i for i, ids in enumerate(token_ids) if ids[0] != config.cls_token_id]
    check(not no_cls,
          f"texts {no_cls} do not start with CLS (id {config.cls_token_id}), so their vectors would pool "
          f"another token")

    models = load_models(base_dir, q8_dir, config)
    tensors = {key: cls_vectors(model, token_ids, batches, config.pad_token_id)
               for key, model in models.items()}
    for kind in ("base", "q8"):
        drift = (tensors[f"{kind}.bf16"] - tensors[f"{kind}.fp32"]).abs().amax(dim=1)
        flat = [i for i, d in enumerate(drift.tolist()) if d == 0]
        check(not flat,
              f"{kind}: texts {flat} show no bfloat16 drift at all, so the checkpoint tests' bounds, "
              f"multiples of that drift, would be zero")
    tensors["base.layers.bf16"] = layer_trace(models["base.bf16"], token_ids[0], config.pad_token_id)
    tensors["q8.layers.bf16"] = layer_trace(models["q8.bf16"], token_ids[0], config.pad_token_id)

    # In float64, as the checkpoint tests compute it.
    cos = torch.nn.functional.cosine_similarity(
        tensors["q8.fp32"].double(), tensors["base.fp32"].double(), dim=1)
    print("  8-bit vs unquantized base, 1 - cosine per text:", [f"{1 - c:.2e}" for c in cos.tolist()])

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "granite-r2-reference.safetensors"))
    meta = {
        "conditions": conditions(),
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "q8_model": Q8_MODEL, "q8_revision": Q8_REVISION,
        "texts": texts, "batches": batches,
        "token_ids": token_ids,
        "trace_positions": TRACE_POSITIONS,
    }
    (out_dir / "granite-r2-reference.json").write_text(
        json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"tier 2 references written to {out_dir}")


def main():
    check(transformers.__version__ == "4.57.3",
          f"transformers {transformers.__version__}: the fixtures and their bounds pin 4.57.3's ModernBERT")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t1 = sub.add_parser("tier1")
    t1.add_argument("--out", type=Path, required=True)
    t2 = sub.add_parser("tier2")
    t2.add_argument("--base", type=Path, required=True)
    t2.add_argument("--q8", type=Path, required=True)
    t2.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "tier1":
        tier1(args.out)
    elif args.command == "tier2":
        tier2(args.base, args.q8, args.out)


if __name__ == "__main__":
    main()
