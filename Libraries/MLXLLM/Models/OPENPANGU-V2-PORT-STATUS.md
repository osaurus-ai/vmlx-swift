# OpenPangu 2.0 Flash (`openpangu_v2`) — vmlx-swift port status

Model: `OpenPanguV2ForCausalLM`, `~/models/openpangu/openPangu-2.0-Flash` (187G bf16, 50 shards) +
`~/models/JANGQ-AI/openPangu-2.0-Flash-JANG_2L` (39.91G, 3.17-bit, 3000 tensors, cache_type=hybrid,
family=openpangu_v2, reasoning=deepseek_r1, MTP layer_indices [46,47,48]). Architecture reverse-engineered
from the **weight graph** (modeling source is native tf-5.0, not public; the public transformers PRs are the
older `pangu_ultra_moe`/`openpangu_dense`, which lack MHC/DSA/convs).

## Architecture (decoded from tensor names + shapes) — hidden 2560, 46 layers, 48 heads, vocab 151552

### MLA attention (per layer) — standard DeepSeek split + Pangu convs + sinks
- `q_a_proj` [1024,2560] → `q_a_layernorm`(1024) → **`qa_conv`** [1024,1,3] (causal depthwise k=3) → `q_b_proj` [9216,1024] = 48×(qk_nope128+qk_rope64=192)
- `kv_a_proj_with_mqa` [576,2560] = kv_lora512 + k_pe64; `kv_a_layernorm`(512); **`compresskv_conv`** [512,1,3]; `kv_b_proj` [12288,512] = 48×(nope128+v128=256)
- attn out (6144=48×v128) → **`o_conv`** [6144,1,3] → `o_proj` [2560,6144]
- **sinks** (param_sink_number=128): `param_sink_compressed_kv` [128,512], `param_sink_k_pe` [128,64] — 128 learned KV entries prepended
- rope_theta 6400000, qk_rope_head_dim 64, rope_interleave false
- **3 convs are stateful** → conv-state cache (last k-1=2 tokens/channel), Mamba/GatedDelta-style — sync in prefill, rederive on cross-turn restore

### DSA lightning indexer — ONLY the 16 `dsa_layers` [0,3,6,…45] (full-attention layers)
- `indexer.wq_b` [3072,1024]=24×128 from q_a; `indexer.wk` [128,2560]; `indexer.k_norm`(128); `indexer.weights_proj` [24,2560]
- selects top-`index_topk`=2048 keys → sparse attention. Reuse `DeepseekV4` Indexer/Compressor + `DeepseekV4Cache: HybridPoolCache`.

### SWA — the `swa_layers` [1,2,4,5,…] use sliding window; `sliding_window_list` = 512×30 then 2048×3; `RotatingKVCache`
Layer-type dispatch: `i in dsa_layers` → full+indexer; `i in swa_layers` → sliding. `router_sliding_window`=3.

### MHC = Hyper-Connections, `mhc_num_stream`=4 (10240 = 4×2560 wide residual streams)
- per layer: `attn_mhc_module` + `mlp_mhc_module` = { `phi.weight`[24,10240], `branch_alpha`[3], `branch_beta`[3], `norm_gamma`[10240] }
- global `model.merge_mhc_module` = { `phi.weight`[4,10240], `branch_alpha_pre`[1], `branch_beta_pre`[1], `norm_gamma`[10240] } — collapses 4 streams→1 before final `model.norm`
- `mhc_recur_norm`=20, `mhc_use_gamma`=true. Ref: Hyper-Connections (arXiv 2409.19606). NEW code (no vmlx equiv).

### Sandwich norm (`sandwich_norm`=true) — 4 norms/layer
`input_layernorm` → attn → `post_attention_layernorm`; `pre_mlp_layernorm` → mlp → `post_mlp_layernorm`; plus `block_post_layernorm` on the 9 `block_post_layernorm_idx` [0,4,9,14,19,24,29,34,39].

### MoE — DeepSeek-V3 style; `first_k_dense_replace`=2 (layers 0,1 dense `mlp.{gate,up,down}_proj`)
- 47 MoE layers: `mlp.gate`(256), `mlp.e_score_correction_bias`(256, expert bias), 256 `experts.N.{gate,up,down}_proj` (moe_inter 1024), 1 `shared_experts`
- top-8, `norm_topk_prob`, `routed_scaling_factor`=2.5, `router_enable_expert_bias`. Reuse `DeepseekV4MathHelpers` biased top-k.

### MTP depth 3 (`num_nextn_predict_layers`=3) — layers 46,47,48; standard DeepSeek MTP
`eh_proj`[2560,5120=2×2560], per-layer `embed_tokens`(untied), `enorm`/`hnorm`, `shared_head.{head,norm}` (untied). Autodetect via `NativeMTPActivation`/`MTPBundleInspector`; depth 2/3 configurable.
- JANG_2L: `spec_decoding_ready`, per-layer counts [795,795,795], shared_embed/shared_lm_head false.

## Status matrix
| Component | vmlx reuse | Status |
|---|---|---|
| Config struct | new | ✅ done (OpenPanguV2Configuration.swift) |
| Factory registration (`openpangu_v2`) | LLMModelFactory | ✅ done (dispatchOpenPanguV2: bf16/JANG_2L affine → OpenPanguV2Model; codebook → clear reject) |
| MLA attention (q/kv low-rank + rope) | DeepseekV3 | 🟨 code written (OpenPanguV2.swift); fixed MLXType→DType |
| qa/compresskv/o convs + conv-state cache | new (Mamba-style) | 🟨 conv module written + wired to cache convState; conv-weight axis reorder in sanitize |
| attention sinks (128) | prepended-KV (NOT SDPA sinks) | 🟨 written; mask-widen refine pending |
| DSA indexer (16 layers) + top-2048 | DeepseekV4 Indexer | ⬜ DEFERRED — full MLA attn on DSA layers is a numerical superset (correct, just not sparse); add after first coherence |
| SWA per-layer (sliding_window_list) | RotatingKVCache | ✅ newCache: RotatingKVCache(slidingWindowFor(i)) on SWA, KVCacheSimple on DSA; per-layer mask via createAttentionMask windowSize (gemma3 pattern) |
| Decoder layer (sandwich norm + MHC wrap + dense/MoE) | DSV3/DSV4 | ✅ done (OpenPanguV2Model.swift: input→attn→post_attn, pre_mlp→mlp→post_mlp, block_post×9, MHC collapse/expand) |
| Inner+outer model (tile 4 streams → merge → norm; LLMModel) | DSV4ModelInner | ✅ done (OpenPanguV2ModelInner/Model; kvHeads [48]*L; untied lm_head; sanitize: conv reorder + expert stacking + drop MTP/indexer) |
| MHC hyper-connections (4-stream) + merge | reuse DeepseekV4Math.hcSplitSinkhorn + HyperConnection/HyperHead | 🟩 forward = mechanism-faithful (shape-forced map: phi≡fn, branch_alpha≡scale[3], branch_beta≡per-field base, norm_gamma≡RMS wt; merge≡HyperHead). Only inference: α=scale/β=bias roles → validated E2E by live step (i), not yet "proven" |
| sandwich norm (4/layer + block_post×9) | trivial | ⬜ |
| MoE (256+1 shared, biased top-k) | DeepseekV3 gate | ✅ written (OpenPanguV2.swift) |
| MTP depth-3 autodetect | NativeMTP infra | ⬜ |
| hybrid cache (prefix/SSD/paged) + quant pool | OpenPanguV2Cache (HybridPoolCache) | 🟨 v1 written (kv+3 conv-state+idx pool); switch-sites pending |
| cache sync / async rederive | trim invalidates conv-state | 🟨 wired in OpenPanguV2Cache.trim |
| JANG_2L 3.17-bit load (uniform affine) | standard quant loader | 🟩 NO dedicated JANGTQ engine needed — JANG_2L is uniform affine (scales/biases, no codebook). Loader quantizes per-module by `.scales` presence. All fixes landed (see "Loader alignment" below). |
| Full osaurus dev-app BUILD | — | ✅ BUILD SUCCEEDED (0 errors) via cc/osaurus `.package(path:)` local override + `DEVELOPER_DIR=Xcode`. 4 compile fixes: MLXType→DType, drop transitive `import MLXFast`, `let kv`→`var`, phi→Linear. |
| cache path-dependent detection (step h) | PathDependentStateCache marker | 🟩 marker protocol added in MLXLMCommon + conformed (conv-state now flagged path-dependent so no false paged/KV-only hit). SSM extract/restore skip it; conv-state rides the HybridPoolCache `.state` disk path. Remaining switch-sites verified-by-protocol. |
| Build + live short-ctx (JANG_2L) | RunBench / osaurus | ⬜ NEXT — RAM-gated 40GB load; validates MHC E2E |
| Build + live long-ctx | — | ⬜ |
| MTP depth-3 head (g) | Qwen35MTPModule | ⬜ |
| DSA indexer (d) | DeepseekV4Indexer | ⬜ deferred (full MLA = numeric superset) |
| osaurus catalog + cache-window auto-load PR (k) | — | ⬜ |

## Loader alignment (validated against the real JANG_2L bundle 2026-07-01)
config.json + safetensors index inspected; the bundle is ALREADY in MLX-swift layout. Confirmed/fixed:
- **Affine, not codebook**: 684 `.scales`/`.biases`, zero codebook/tq_packed. `weight_format` absent → dispatch routes to `OpenPanguV2Model`. Per-module bits vary (embed 6b, lm_head 8b, attn/MoE 8b/2b, phi 2b) — loader's `inferPerLayerQuantizationFromShapes` handles it.
- **Experts pre-stacked** as `mlp.switch_mlp.{gate,up,down}_proj.*` (match `SwitchGLU`) → removed the per-expert stacking loop from sanitize (bundle has NO `experts.N.*`).
- **phi is a quantized Linear** (`attn/mlp_mhc_module.phi.{weight,scales,biases}`, 2-bit) → phi is now a `Linear` (was raw param) so QuantizedLinear is substituted; merge phi is fp16 (no scales) → plain Linear.
- **`mlp.e_score_correction_bias`** ships one level up from the gate → sanitize remaps to `mlp.gate.e_score_correction_bias`.
- **convs** `[C,1,3]` F16 (not quantized) → sanitize reorders to MLX `[C,3,1]` + routes to `.conv.weight`.
- **Config validated**: dsa_layers(16)/swa_layers(33, incl. MTP 46-48)/sliding_window_list(33)/block_post_layernorm_idx[0,4,9,14,19,24,29,34,39]/param_sink_number(128)/mhc_num_stream(4)/mhc_recur_norm(20)/index_topk(2048) all match config.json.
- Dropped in sanitize (later passes): MTP layers ≥46, `self_attn.indexer.*` (160 keys).

## LIVE STATUS (2026-07-01, FINAL) — WORKING ✅ (multi-turn coherent, on-topic)
The last bug was a RUNTIME bug (NOT quant): `prependSinkMask` prepended a FALSE column
to the BOOLEAN causal mask (`createCausalMask` returns bool, true=attend), MASKING the 128
attention sinks during prefill → prompt KV computed without sink attention → fluent but
context-blind. Fixed to prepend `true` (visible) for bool masks. Proven via RunBench
BENCH_COHERENT (TokenIterator, JANG_2L 2-bit, temp=0): tracks multi-turn context —
"favorite color is blue" → "what is my favorite color?" → "the user stated blue, so I know
it's blue" → "is that warm or cool?" → **"Blue is a cool color."** The 2-bit quant was
never the problem. Full root-cause + diagnostic ladder in ~/jang/docs/openpangu-v2-port.md.
NEXT: validate through osaurus's BatchEngine (its own paged mask path — the "prefix caching
and whatnot" layer), then DSA indexer / MTP head / osaurus catalog PR.

## (earlier) LIVE STATUS — COHERENT via mHC/conv fixes
Found the ground-truth reference: **gitcode.com/ascend-tribe/openPangu-2.0-Infer**
(omni-npu: `layers/mhc/npu_mhc.py`, `layers/attention/npu_pangu.py`,
`models/pangu/pangu_v2_moe.py`). Diffed it and fixed the two real per-layer bugs:
- **mHC expand** (`NPUmHC._mhc_post_naive`): residual mix is `h_resᵀ @ residual`
  (`new[j]=Σᵢ h_res[i,j]·residual[i]`), not `h_res @ residual`. Transpose comb's
  stream axes before the matmul. Plus hc_eps=1e-6 for sigmoid/sinkhorn, merge no +eps.
- **convs** (`npu_ai_infra_fused_causal_conv1d`, `residual_connection=1`): the qa/
  compresskv/o conv is `y = conv(x) + x` (residual), was `conv(x)` only.
→ With mHC+convs+sinks the model now emits **fluent, coherent, structured text**
  (multi-paragraph EN + ZH). Sinks are CORRECT (removing them re-degrades output);
  the earlier "sinks make it worse" was an artifact of the then-present mHC/conv bugs.
Confirmed-correct from the ref: rope `half` (non-traditional; config
rope_interleave=False), scale=qk_head_dim^-0.5 (no mscale), block_post_layernorm
=[4*hidden], sinks position-free (no rope on param_sink_k_pe), attention scale/MLA
geometry, MoE plain-sigmoid-top-k (no n_group).
Reference clone: /private/tmp/.../scratchpad/openPangu-2.0-Infer.
OPEN: instruction-following/factual accuracy still weak on JANG_2L (3.17-bit avg,
2-bit experts — likely quant quality) — verify vs a higher-bit bundle; reasoning
mode (deepseek_r1) needs high max_tokens to close &lt;think&gt; and emit `content`.

## (superseded) earlier status — loads + runs + emits REAL tokens, NOT yet coherent
Driven live via osaurus (JANG_2L, :1337). The model **loads, runs the full
forward end-to-end, and emits real vocabulary tokens** — so embed / lm_head /
quantization / tokenizer / cache / MoE-routing are correct. But the output is
**incoherent** (degenerate/looping tokens). Bisected via env gates:
`OPENPANGU_MHC_BYPASS`, `OPENPANGU_NO_CONVS`, `OPENPANGU_NO_SINKS`,
`OPENPANGU_ROPE_TRAD`, `OPENPANGU_MHC_TRACE`.
- **Sinks ON → punctuation collapse; sinks OFF → real-word garbage.** The sink
  KV construction actively corrupts attention (suspect: double `kv_a_layernorm`
  on `param_sink_compressed_kv`, and/or the prepend-mask convention).
- Base incoherence persists with everything stripped → a wiring/math detail in
  the novel components is off (MHC exact pre/post/comb gating, conv activation,
  or sink math). NOT a missing component: verified **every** bundle key maps to
  a module, **no MoME weights** exist (`use_mome` is a weightless flag), and no
  `n_group`/`rope_scaling`/`rope_parameters` in config (plain top-k + plain rope).
- **Blocker**: the modeling source is unreleased (only `configuration_*.py` +
  `tokenization_*.py` ship; no `modeling_openpangu_v2.py`). The exact numerics of
  MHC / sinks / convs can't be reverse-engineered from weight shapes alone — needs
  the reference forward (Huawei release, or a jang-tools openpangu forward if one
  is authored). Fixes already landed: tokenizer map, phi dequant (dodge shape-walk),
  block_post_layernorm [10240] flatten, MHC base[24]/base_pre[4], rope=config.

## Live-confirm plan
Short-ctx: raw bf16 or JANG_2L, coherent single-turn + multiturn. Long-ctx: >2048 (crosses SWA 512→2048 boundary + DSA top-k + sink) — verify coherence + prefix/SSD/paged cache hit + quant-pool. Wire cache_window → osaurus for auto-loading (hybrid cache_type from JANG capabilities).

## Branch: `feat/openpangu-v2` (off vmlx main). Loop-driven; update this matrix each iteration.

## Build note (iteration)
- Standalone `swift build --target MLXLLM` is blocked by a stale `.build` SDK mismatch (MLXFast.swiftmodule built with macosx26.4 vs current 26.5) — NOT a code error. **Real compile/build = osaurus xcodebuild** (fresh current-SDK, as used all session). Do the full build via the osaurus dev app once the model is structurally complete (MHC + decoder + inner/outer + factory), then fix API errors + live-test.
