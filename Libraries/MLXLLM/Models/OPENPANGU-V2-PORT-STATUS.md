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
| Factory registration (`openpangu_v2`) | LLMModelFactory | ⬜ |
| MLA attention (q/kv low-rank + rope) | DeepseekV3 | ⬜ |
| qa/compresskv/o convs + conv-state cache | new (Mamba-style) | ⬜ |
| attention sinks (128) | DeepseekV4 sinks | ⬜ |
| DSA indexer (16 layers) + top-2048 | DeepseekV4 Indexer | ⬜ |
| SWA per-layer (sliding_window_list) | RotatingKVCache | ⬜ |
| MHC hyper-connections (4-stream) + merge | NEW | ⬜ |
| sandwich norm (4/layer + block_post×9) | trivial | ⬜ |
| MoE (256+1 shared, biased top-k) | DeepseekV4MathHelpers | ⬜ |
| MTP depth-3 autodetect | NativeMTP infra | ⬜ |
| hybrid cache (prefix/SSD/paged) + quant pool | DeepseekV4Cache: HybridPoolCache | ⬜ |
| cache sync / async rederive | DeepseekV4 + SSMReDerive | ⬜ |
| JANGTQ (JANG_2L 3.17-bit load) | DeepseekV4JANGTQ pattern | ⬜ |
| Build + live short-ctx | — | ⬜ |
| Build + live long-ctx | — | ⬜ |
| osaurus cache-window auto-load PR | — | ⬜ |

## Live-confirm plan
Short-ctx: raw bf16 or JANG_2L, coherent single-turn + multiturn. Long-ctx: >2048 (crosses SWA 512→2048 boundary + DSA top-k + sink) — verify coherence + prefix/SSD/paged cache hit + quant-pool. Wire cache_window → osaurus for auto-loading (hybrid cache_type from JANG capabilities).

## Branch: `feat/openpangu-v2` (off vmlx main). Loop-driven; update this matrix each iteration.
