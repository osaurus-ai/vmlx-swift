# Muse Glimmer 30B — architecture spec

Reference: `meta-models/Muse-Glimmer-30B` (`model_type: muse_glimmer`,
`MuseGlimmerForConditionalGeneration`, transformers 5.15.0.dev0).
Local bundles: `~/models/meta-models/Muse-Glimmer-30B` (bf16, 55G),
`~/models/JANGQ-AI/Muse-Glimmer-30B-JANG_6M` (26G, 6-bit gs64 affine),
`~/models/JANGQ-AI/Muse-Glimmer-30B-JANG_4M` (20G, 4-bit gs64 affine).
All four bundles verified complete (no truncated shards).

Dense 30B image+video VLM. **Not** a Gemma re-skin despite the Gemma2 config
parent — five details below diverge and each one produces coherent-but-wrong
output if missed.

## Text tower — 52 dense layers

| field | value |
|---|---|
| hidden / intermediate | 6656 / 19968 |
| heads / kv heads / head_dim | 32 / 2 / 128 (GQA, no bias) |
| vocab / tie | 202048 / **not tied** |
| rms_norm_eps / post_norm_eps | 1e-5 / **1e-8** |
| sliding_window | 2048 |
| layer_types | 39 `sliding_attention`, 13 `full_attention`, pattern `[w,w,w,full]` |
| layer_rope_theta | 500000 on sliding layers, **0 (NoPE)** on full layers |
| qk_scale_factor | 3.87 |
| output_multiplier | 0.19611613513818404 = `1/sqrt(hidden/256)` |
| final_logit_softcapping | 20.0 |

### The five divergences

1. **`CenteredRMSNorm`** — `x * rsqrt(mean(x²)+eps) * (1.0 + weight)`. Weights
   are stored zero-centered, so the `1.0 +` is mandatory. Missing it does not
   crash; it degrades output. All four layer norms and the final norm use it.

2. **Scaleless QK-norm + asymmetric scale** — `qk_norm` is an RMSNorm with **no
   learned weight** (unit gain), applied per-head over `head_dim` to Q and K
   after projection/reshape, before RoPE:
   ```
   q = qk_norm(q) * 3.87      # qk_scale_factor applied to Q ONLY
   k = qk_norm(k)             # K gets the norm but not the factor
   ```
   Attention `scale` stays the standard `head_dim ** -0.5`; the 3.87 is on top.

3. **NoPE on full-attention layers** — `layer_rope_theta[i] == 0` means that
   layer receives no rotary at all. Every 4th layer counted backward from the
   last (`(52-1-i) % 4 == 0` → i = 3, 7, …, 51). These are exactly the
   `full_attention` layers; sliding layers get theta 500000.

4. **Attention output gate** — a `gate_proj` (hidden → heads*head_dim, no bias)
   reading the *layer input* (post-`input_layernorm` hidden states), applied
   **before** `o_proj`:
   ```
   attn = attn.reshape(B, L, heads*head_dim) * sigmoid(gate_proj(hidden_states))
   out  = o_proj(attn)
   ```
   Note this is `self_attn.gate_proj` — distinct from `mlp.gate_proj`.

5. **Logit tail** — `logits = lm_head(h) * output_multiplier`, then Gemma-style
   `20 * tanh(logits / 20)`. **No embedding normalizer**: unlike Gemma2 the
   input embeddings are *not* scaled by `sqrt(hidden_size)`.

### Layer order (sandwich norms, Gemma2-style)

```
r = h;  h = input_layernorm(h)            # eps 1e-5
h = self_attn(h)
h = post_attention_layernorm(h)           # eps 1e-8, BEFORE the residual add
h = r + h
r = h;  h = pre_feedforward_layernorm(h)  # eps 1e-5
h = down(silu(gate(h)) * up(h))
h = post_feedforward_layernorm(h)         # eps 1e-8, BEFORE the residual add
h = r + h
```

## Vision tower — 50 layers, image **and video**

Qwen2-VL-shaped: windowed attention with `cu_seqlens`, `grid_thw`, patch merge.

| field | value |
|---|---|
| hidden / intermediate / heads | 1536 / 8960 / 16 |
| patch_size / merge_size | 14 / 2 |
| **patch_temporal** | **2** — video arrives as 2-frame patch pairs |
| layer_types | `[win,win,win,full]` × 12 + `[win,full]`, 50 total |
| pos emb | learned table 32×32 (`pos_emb_height/width`), interpolated |
| rope_theta | 10000 |

LayerNorm (not RMSNorm) with bias: `norm1`/`norm2`, plus `ln_pre`/`ln_post`.
Attention has q/k/v/proj **with bias**. Pre-norm residual:
`h += attn(norm1(h)); h += mlp(norm2(h))`.

### Confirmed pipeline (read off the JANG_6M weight shapes, not the config)

```
pixels → patch_embedding      [1536, 1176]   1176 = 14*14 * 2(temporal) * 3(ch), no bias
      + position_embedding_table [1024, 1536]  learned 32*32 grid
      → ln_pre                 LayerNorm(1536) with bias
      → 50 × block             attn q/k/v/proj [1536,1536] all WITH bias,
                               norm1/norm2 LayerNorm(1536) with bias,
                               mlp fc1 [8960,1536] / fc2 [1536,8960] with bias,
                               pre-norm residual: h += attn(norm1(h)); h += mlp(norm2(h))
      → ln_post                LayerNorm(1536) with bias
      → spatial merge 2×2      4 × 1536 = 6144  ← this is where out_hidden_size comes from
      → vision_adapter.fc1     [4096, 6144] no bias → gelu
      → vision_adapter.fc2     [4096, 4096] no bias → gelu   (act applied TWICE:
                                                              act(fc2(act(fc1(x)))))
      → vision_projection      [6656, 4096] → text width 6656
      → scatter into the embedding sequence at <|patch|> / <|video|> positions
```

Vision attention is **non-causal**, 16 heads → head_dim 96, windowed via
`cu_seqlens` with every 4th layer full — the Qwen2-VL shape, so
`QwenVL.VisionRotaryEmbedding`, `QwenVL.PatchEmbed`, `QwenVL.patchify` and
`QwenVL.mergeInputIdsWithImageFeatures` are the right building blocks.

The vision tower is **not quantized** in the JANG bundles — every vision tensor
is plain F16 with no `.scales`/`.biases` siblings. A loader that assumes the
whole checkpoint is quantized will look for scales that do not exist.

Placeholders: `image_token_id` 200092 (`<|patch|>`), `video_token_id` 200091
(`<|video|>`); features are scattered into the embedding sequence at those
positions.

## Chat template — "Onyx ATEM"

Special tokens: `<|start|>`, `<|message|>`, `<|eot|>` (end turn), `<|eom|>`
(end message, used when the assistant continues), `<|patch|>`, `<|video|>`.
Generation prompt is `<|start|>assistant` with **no** trailing `<|message|>`.

**Tool calls use a new XML dialect, not JSON:**

```
<atem:function_calls>
<atem:invoke name="FUNCTION_NAME">
<atem:parameter name="PARAM">value</atem:parameter>
</atem:invoke>
</atem:function_calls>
```

Scalars are written bare; lists/objects/booleans as JSON. Multiple `<atem:invoke>`
blocks may appear inside one `<atem:function_calls>`. The template itself says
the output "is not expected to be valid XML and is parsed with regular
expressions" — so the parser must be tolerant, and must survive a stray `<`
(cf. the gemma4 stray-angle desync).

**Reasoning is a system-prefix string, not a channel**: `render_reasoning()`
emits `Reasoning strength: high|medium|low.` into the system message from the
`reasoning_strength` template variable (default `high`). There are no
`<think>` tags and no reasoning channel.

### Cache consequence (important)

Because reasoning strength is rendered **into the system prefix**, changing
effort mutates token ~N of the prompt and invalidates the entire prefix — every
boundary rung, not just the tail. This is the strongest form of the
"changed reasoning effort → cache miss" case:

- The scope salt (`reasoning=…|effort=…`) must cover `reasoning_strength`, or
  two different prefixes collide on one cache key and produce wrong reuse.
- Salting correctly still means a full re-prefill on every effort flip, so each
  effort deserves its own boundary ladder rather than fighting for one.
- Tool availability changes shift the same prefix (tool defs render in the
  system message too), so tool-toggle and effort-flip are the same failure
  class here.
