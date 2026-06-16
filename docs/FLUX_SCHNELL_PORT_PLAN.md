# FLUX.1-schnell native port plan (vMLXFlux)

Concrete plan to take `flux1-schnell` from `notImplemented` to a live-proven native
pipeline. Reference: the mflux Python source (`mflux/models/flux/...`), studied
2026-06-15. Template: the proven `ZImageNative.swift` + the existing `FluxDiT.swift`
(transformer topology + 40-line BFL weight-key docblock) + `VAE.swift` (Flux VAE).

**Test bundle:** `dhairyashil/FLUX.1-schnell-mflux-4bit` (9.6 GB, staged to
`~/.mlxstudio/models/image/`). Layout: `text_encoder/` (CLIP-L, 69 MB),
`text_encoder_2/` (T5-XXL, 2.7 GB ×2 shards), `tokenizer/`+`tokenizer_2/`,
`transformer/` (4 shards, ~6.7 GB), `vae/` (165 MB). Also `-8bit` available for the
quant matrix; full precision = `black-forest-labs/FLUX.1-schnell`.

## Generation flow (from mflux `variants/txt2img/flux.py`)
1. Initial latents = flow-match noise (have: `LatentSpace`/`FlowMatchScheduler`).
2. Encode prompt: **T5** → `prompt_embeds` (per-token, the main conditioning);
   **CLIP** → `pooled_prompt_embeds` (pooled vector → transformer `vector_in`).
3. For each step: `noise = transformer(t, latents, prompt_embeds, pooled_prompt_embeds)`;
   `latents = scheduler.step(noise, t, latents)`; `mx.eval(latents)` for progress.
4. `unpack_latents` → `VAE.decode` → image. Schnell: 4 steps, guidance 0 (no CFG).

## Pieces to port (with concrete arch)

### A. T5-XXL encoder — `vMLXFluxKit/TextEncoders/T5XXL.swift` (NEW, ~250 lines)
EXACT spec (from mflux `t5_encoder/`, dims verified 2026-06-15):
- `shared`: Embedding(32128, 4096).
- 24 × T5Block = { T5Attention, T5FeedForward }, each sub-layer is residual:
  - **T5Attention**: `h = h + SelfAttention(T5LayerNorm(h))`.
  - **T5SelfAttention**: q,k,v,o = Linear(4096→4096, bias=false). **64 heads × 64
    head_dim**. shape: (1,seq,4096)→reshape(1,seq,64,64)→transpose(0,2,1,3)=(1,64,seq,64).
    `scores = q @ kᵀ` with **NO 1/sqrt(d) scaling**. Add `position_bias`. softmax. `@ v`.
    un_shape → o(·).
  - **relative_attention_bias**: Embedding(32, 64) PER block (each T5SelfAttention has
    its own — verify against weight map whether checkpoint shares or per-layer).
    bias = embed(bucket(mem_pos − ctx_pos)) → transpose(2,0,1) → expand_dims(0).
    Bucketing: bidirectional, num_buckets=32, max_distance=128:
    `bucket = (rel>0 ? 16 : 0)`; `rel=abs(rel)`; `is_small = rel<8`;
    `large = 8 + floor(log(rel/8)/log(128/8)*(16−8))` clipped to ≤15;
    `bucket += is_small ? rel : large`.
  - **T5LayerNorm**: RMSNorm, no mean-subtract, no bias. `x*rsqrt(mean(x²)+1e-6)*weight`.
  - **T5FeedForward**: `h = h + DenseReluDense(T5LayerNorm(h))`.
  - **T5DenseReluDense**: wi_0(4096→10240), wi_1(4096→10240), wo(10240→4096), all
    bias=false. `out = wo( new_gelu(wi_0(x)) * wi_1(x) )`.
    new_gelu = `0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³)))` (tanh-approx GELU).
- `final_layer_norm` = T5LayerNorm. Output: (1, seq, 4096) per-token = prompt_embeds.
- Shared across flux1/flux2/wan — biggest single payoff; put in Kit. (Qwen uses
  Qwen2.5-VL, NOT T5.)

### B. CLIP-L encoder — `vMLXFluxKit/TextEncoders/CLIPText.swift` (NEW, ~150 lines)
From `clip_encoder/`: `CLIPTextModel(dims=768, num_encoder_layers=12)` → pooled
output (take the EOS-token hidden state). Standard pre-LN CLIP transformer +
position embeddings. Output: pooled 768-vec.

### C. Transformer wiring — extend `FluxDiT.swift` / write `Flux1Native.swift`
`FluxDiT.swift` already has `FluxDoubleStreamBlock` / `FluxSingleStreamBlock` /
`FluxDiTConfig.schnell` (19+38 blocks). Needs:
- Real weight load via the key-map (`flux_weight_mapping.py`, 817 lines — the source
  of truth; map BFL keys → Swift module paths). Either an explicit remap table or
  `@ModuleInfo(key:)` decorators.
- **3-axis RoPE** over (text-pos=0, H, W) — currently TODO in the double/single
  blocks. `embed_nd.py` in mflux has the position-id construction.
- `img_in`/`txt_in`/`time/vector/guidance` projections; for schnell guidance is unused.

### D. mflux 4-bit dequant
The `*-mflux-4bit` transformer/T5 linears are packed with scale tensors exactly like
z-image's — reuse `ZImageWeightStore`'s scale-tensor dequant `linear(...)`.

## Live-proof gate (do NOT mark done before this)
`vmlxflux-probe --model FLUX.1-schnell-mflux-4bit --generate --seed S --steps 4
--turn A --turn B --turn A`:
- turns 1≡3 byte-identical (determinism); turn 2 ≠ 1 (prompt sensitivity);
- both coherent + prompt-accurate; compare vs the repo's `comparison.png`.
Then 8-bit (`dhairyashil/...-8bit`) + full precision → same matrix. Update
`OSAURUS_VMLX_FLUX_INTEGRATION_SPEC.md` §6/§11 + the API spec's model list.

## GROUNDED checkpoint facts (from the staged dhairyashil/...-4bit, mflux 0.6.2)
- **Quant = standard MLX group-quant** (`weight`/`scales`/`biases`, group 64, bits 4).
  Use the shared `MFluxStore`/`MFluxLinear`/`MFluxEmbedding` (NEW, `Common/MFluxQuant.swift`,
  compiles) — same format z-image-4bit uses. Plain params (layernorms, conv bias) load as-is.
- **CLIP (`text_encoder/`)** keys: `text_model.embeddings.{token_embedding,position_embedding}`,
  `text_model.encoder.layers.{0..11}.{layer_norm1,layer_norm2,mlp.{fc1,fc2},self_attn.{q,k,v,out}_proj}`,
  `text_model.final_layer_norm`. 12 layers, dim 768, linears HAVE bias. CAUSAL mask.
  pooled = `last_hidden_state[0, argmax(tokens)]` (EOS-token row) → `final_layer_norm`.
- **T5 (`text_encoder_2/`)** keys: `shared` (emb), `t5_blocks.{0..23}.attention.{SelfAttention.{q,k,v,o,
  relative_attention_bias}, layer_norm}`, `t5_blocks.N.ff.{DenseReluDense.{wi_0,wi_1,wo}, layer_norm}`,
  `final_layer_norm`. relative_attention_bias IS per-block (all 24 present). Matches §A spec.
- **Transformer (`transformer/`)** keys: `context_embedder`, `x_embedder`(check), `time_text_embed.*`,
  `transformer_blocks.{0..18}` (19 double), `single_transformer_blocks.{0..37}` (38 single,
  `attn.{norm_q,norm_k,to_q,to_k,to_v}`, `norm.linear`, `proj_mlp`, `proj_out`), `norm_out.linear`,
  `proj_out`. Matches FluxDiT.swift topology — map these exact keys.
- **VAE (`vae/`)** keys: `decoder.{conv_in.conv2d, mid_block, up_blocks.*, conv_norm_out.norm, conv_out.conv2d}`
  — standard AutoencoderKL (VAE.swift has it).
- **3-axis RoPE** (`embed_nd.py`): `EmbedND` dim 3072, theta 10000, `axes_dim=[16,56,56]` over
  ids (txt-pos=0 for all text tokens; image tokens get (0, row, col)). Build cos/sin per axis,
  concat on head-dim, apply to q/k. This is the currently-TODO RoPE in FluxDiT blocks.

## Progress
- [x] `Common/MFluxQuant.swift` — shared MLX-quant store + layers (compiles).
- [x] `Common/T5XXL.swift` — T5-XXL encoder (24 blocks, 64h×64d, rel-pos-bias bucketing,
  gated-GELU FFN), compiles. Component "text_encoder_2".
- [x] `Common/CLIPText.swift` — CLIP-L encoder (12 layers, causal SDPA 12h×64, quick-gelu MLP,
  pooled=EOS-token row), compiles. Component "text_encoder".
- [x] `Flux1/Flux1Native.swift` — FULL DiT transformer COMPILES: FluxRoPE(EmbedND apply),
  FluxTimeTextEmbed, FluxAdaNormZero/Single/Continuous, FluxFeedForward, FluxJointAttention(24h×128)
  + FluxSingleAttention, FluxJointBlock×19, FluxSingleBlock×38, FluxTransformer assembler
  (x_embedder/context_embedder/norm_out/proj_out + ropeFreqs latent ids).
- [ ] VAE decoder for flux keys (`decoder.conv_in.conv2d`, mid_block, up_blocks, conv_out.conv2d) —
  AutoencoderKL; adapt ZImageVAEDecoder pattern. Flux VAE scale/shift: latent/0.3611+0.1159 (verify).
- [ ] tokenizers (CLIP `tokenizer/` max77, T5 `tokenizer_2/` max256) via AutoTokenizer.from(modelFolder:).
- [ ] `FluxSchnellPipeline` in Flux1Native: load weights→T5XXLEncoder+CLIPTextEncoder+FluxTransformer+VAE;
  generate = noise(1,hw,64)→loop[transformer→FlowMatch euler]→unpack→VAE decode→PNG. Wire Flux1Schnell.
- [x] VAE decoder (FluxVAEDecoder, conv2d keys) + tokenizers (FluxTokenizers, needs tokenizer.json —
  see note) + FluxSchnellPipeline + Flux1Schnell.generate wired.
- [x] **LIVE-PROVEN 2026-06-15 (FLUX.1-schnell-mflux-4bit, 512px/4-step, ~3.9s/img):** all stages finite
  (T5 [1,256,4096], CLIP [1,768], transformer [1,1024,64], image [1,3,512,512]); coherent +
  prompt-accurate (photo red apple; watercolor snowy mountain w/ correct style); determinism
  turn1≡turn3 byte-identical (sha d3ed219d); prompt-sensitivity turn2≠turn1. WORKS on first real run.

**TOKENIZER NOTE (important for osaurus):** the mflux-4bit bundle ships SLOW tokenizer files only
(CLIP vocab.json+merges.txt, T5 spiece.model) but swift-transformers AutoTokenizer.from(modelFolder:)
needs `tokenizer.json` (fast). Convert once with `transformers` (AutoTokenizer.from_pretrained(dir,
use_fast=True).save_pretrained(dir)) → writes tokenizer.json. osaurus staging must ensure tokenizer.json
exists (convert on download, or bundle it). Done for the staged 4bit model.

- [ ] 8-bit (dhairyashil/...-8bit) + full precision quant matrix · [ ] then commit/push.

## TRANSFORMER SPEC (complete — transcribe to Flux1Native.swift, component "transformer")
All linears via MFluxLinear (most quantized; norms affine=false have NO weights).
Dims: model 3072, attn 24 heads × 128, joint/single counts 19/38.

**Latent (FluxLatentCreator):** noise = normal (1, (h//16)*(w//16), 64) seeded.
pack: (1,16,h//16,2,w//16,2)→transpose(0,2,4,1,3,5)→(1, hw, 64).
unpack: (1,h//16,w//16,16,2,2)→transpose(0,3,1,4,2,5)→(1,16,h//8,w//8).

**Forward(t, latents(1,hw,64), prompt_embeds(1,Tt,4096), pooled(1,768)):**
1. `hidden = x_embedder(latents)` Linear(64→3072, bias).
2. `enc = context_embedder(prompt_embeds)` Linear(4096→3072, bias).
3. `text_emb = TimeTextEmbed(timestep, pooled, guidance)` (see below) → (1,3072).
4. `rope = EmbedND(ids)` where ids = concat(txt_ids, img_ids) on axis1:
   txt_ids = zeros(1, Tt, 3); img_ids = zeros(h//16, w//16, 3) with [...,1]+=row, [...,2]+=col,
   reshaped (1, (h//16)(w//16), 3). EmbedND: dim3072, θ1e4, axes_dim[16,56,56]; per axis i
   rope(pos=ids[...,i], axes_dim[i]): scale=arange(0,d,2)/d, omega=1/θ^scale, out=pos⊗omega,
   stack[cos,-sin,sin,cos]→reshape(B,seq,d//2,2,2); concat 3 axes on axis=-3; expand_dims(axis=1)
   → (1,1, Tt+hw, 64, 2, 2).
5. for 19 JointTransformerBlock: (enc, hidden) = block(hidden, enc, text_emb, rope).
6. `hidden = concat([enc, hidden], axis=1)`.
7. for 38 SingleTransformerBlock: hidden = block(hidden, text_emb, rope).
8. `hidden = hidden[:, Tt:, :]` (drop text); `hidden = norm_out(hidden, text_emb)`; `proj_out` Linear(3072→64, bias). Return (1, hw, 64).

**TimeTextEmbed** (`time_text_embed.*`): timestep_embedder = Linear(256→3072,bias)→silu→Linear(3072→3072,bias)
on time_proj(timestep); text_embedder = Linear(768→3072,bias)→silu→Linear(3072→3072,bias) on pooled;
conditioning = time_emb + text_emb. (schnell: guidance_embedder = None.)
time_proj(t): half=128, max_period 1e4, exponent=-ln(1e4)*arange(128)/128, emb=exp(exponent);
e=t[:,None]*emb[None]; concat[sin(e),cos(e)]; then SWAP halves: concat[e[:,128:], e[:,:128]] → (1,256).
timestep = sigmas[step]*1000 (num_train_steps=1000).

**JointTransformerBlock** (`transformer_blocks.N.`): norm1, norm1_context = AdaLayerNormZero
(linear 3072→18432,bias; silu(text_emb)→6 chunks of 3072: shift_msa,scale_msa,gate_msa,shift_mlp,
scale_mlp,gate_mlp; out = LN_affineFalse(h)*(1+scale_msa[:,None])+shift_msa[:,None]).
attn = JointAttention. norm2/norm2_context = LayerNorm affine=false (normalize last axis, no weights).
ff = FeedForward(gelu), ff_context = FeedForward(gelu_approx); FeedForward = linear1(3072→12288,bias)
→act→linear2(12288→3072,bias).
flow: (nh,gmsa,smlp,scmlp,gmlp)=norm1(h); (nenc,...)=norm1_context(enc);
(attn_out, ctx_attn_out)=attn(nh,nenc,rope);
apply_norm_ff(h, attn_out, gmlp,gmsa,scmlp,smlp, norm2, ff):
  attn_out = gmsa[:,None]*attn_out; h=h+attn_out; nh2=norm2(h); nh2=nh2*(1+scmlp[:,None])+smlp[:,None];
  ff_out=gmlp[:,None]*ff(nh2); h=h+ff_out. Same for enc with _context modules.

**JointAttention** (`transformer_blocks.N.attn.`): 24h×128. to_q/to_k/to_v/to_out.0,
add_q_proj/add_k_proj/add_v_proj/to_add_out (all Linear 3072→3072,bias),
norm_q/norm_k/norm_added_q/norm_added_k = RMSNorm(128) (weights present).
process_qkv(h, q,k,v, nq,nk): q=to_q(h) reshape(1,S,24,128) transpose(0,2,1,3); norm in fp32; (k,v same, v NOT normed).
img q,k,v from hidden; txt from encoder via add_* / norm_added_*.
concat [txt, img] on axis=2 (seq) for q,k,v. apply_rope(q,k,rope). SDPA scale 1/sqrt(128).
transpose+reshape → (1, S, 3072). split [:Tt]=enc, [Tt:]=img. img→to_out[0], enc→to_add_out.

**SingleTransformerBlock** (`single_transformer_blocks.N.`): norm = AdaLayerNormZeroSingle
(linear 3072→9216,bias; silu(text)→3 chunks 3072: shift,scale,gate; out=LN_affFalse(h)*(1+scale)+shift, returns (h,gate)).
attn = SingleBlockAttention (to_q/to_k/to_v 3072→3072,bias; norm_q/norm_k RMSNorm128; process_qkv+rope+SDPA; NO to_out).
proj_mlp = Linear(3072→12288,bias); proj_out = Linear(15360→3072,bias).
flow: residual=h; (nh,gate)=norm(h,text); attn_out=attn(nh,rope);
ff=gelu_approx(proj_mlp(nh)); cat[attn_out, ff] axis2 (→15360); h=gate[:,None]*proj_out(cat); return residual+h.

**norm_out** = AdaLayerNormContinuous(`norm_out.`): linear(3072→6144, NO bias); silu(text)→2 chunks
3072: scale THEN shift (order differs from zero!); out = LN_affFalse(x)*(1+scale)[:,None,:]+shift[:,None,:].

**apply_rope(xq,xk,freqs(...,2,2))**: x_=x.reshape(...,-1,1,2); out=freqs[...,0]*x_[...,0]+freqs[...,1]*x_[...,1]; reshape back to x.shape. (fp32.)

**VAE**: keys `decoder.{conv_in.conv2d, mid_block.{resnets,attentions}, up_blocks.N.{resnets,upsamplers}, conv_norm_out.norm, conv_out.conv2d}`. AutoencoderKL family (same as z-image's ZImageVAEDecoder — adapt key names conv→conv2d). preprocess: latent = latent/0.3611 + 0.1159 then decode (CHECK mflux flux_vae/vae.py scale/shift).

**Tokenizers**: CLIP from `tokenizer/` (max_len 77), T5 from `tokenizer_2/` (max_len 256, pad). via AutoTokenizer.from(modelFolder:).

## Risk / sequencing
- T5 relative-position-bias bucketing is the subtle bit; verify T5 output against a
  Python mflux dump of `t5_text_encoder(input_ids)` for one prompt before trusting.
- Order: T5 (shared) → CLIP → transformer weight-map + RoPE → end-to-end → prove.
- After flux1-schnell proves, flux1-dev is the same + guidance embed; qwen-image
  reuses T5? No — qwen uses Qwen2.5-VL (see QWEN_IMAGE_PORT_PLAN.md), so flux and
  qwen text encoders are independent.
