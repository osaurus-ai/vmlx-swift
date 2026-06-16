# Qwen-Image / Qwen-Image-Edit native port plan (vMLXFlux)

**Purpose:** a concrete, executable port record for `qwen-image` and
`qwen-image-edit`, using the **already-proven `ZImageNative.swift`** as the
reference template. For osaurus teammates + future porting sessions.

**Status (2026-06-16):** `qwen-image` text-to-image is implemented in
`Common/QwenImageNative.swift` and live-proven for the local 4-bit, 6-bit, and 8-bit
mflux bundles with same-seed determinism, prompt sensitivity, and visual
inspection. In the Osaurus monorepo worktree, `qwen-image-mflux-4bit` also has
fresh load proof at
`docs/local/vmlx-flux-probes/2026-06-16-osaurus-qwen-image-q4-load-final/qwen-image-mflux-4bit-load.json`.
`Qwen-Image-mflux-6bit` is staged locally from `filipstrand/Qwen-Image-mflux-6bit`
and live-proven at
`docs/local/vmlx-flux-probes/2026-06-16-qwen-image-6bit-gen20-after-key-fix/Qwen-Image-mflux-6bit-load.json`;
turns 1 and 3 share apple SHA
`66e8187e887087e8a8e9227a99f16236c5ba15717a5e31e08a5772868b3a456a`, while the
blue watercolor mountain prompt has SHA
`44069312716932d6d72181a808625a33777ed29af7723eea8f76b0ac5ba96a52`. The 6-bit
bundle uses Diffusers-style nested modulation keys
`img_norm1.mod_linear` / `txt_norm1.mod_linear`; the native loader accepts both
that layout and the 4-bit `img_mod_linear` / `txt_mod_linear` layout.
`qwen-image-mflux-8bit` is staged locally from `AbstractFramework/qwen-image-8bit`
(repo SHA `c92c2b346b705f797897a29c4a7b5334cb501f44`; index metadata
`quantization_level=8`, `mflux_version=0.18.2`) and live-proven at
`docs/local/vmlx-flux-probes/2026-06-16-qwen-image-8bit-gen20/qwen-image-mflux-8bit-load.json`;
turns 1 and 3 share apple SHA
`1611715920c71fbb40c02011035ff616c93745386e2708570c595f439b15cae3`, while the
blue mountain/sun watercolor prompt has SHA
`67a9f6195777c06a70e81230756015718110472df38ef5688608516637862c70`. The 8-bit
bundle quantizes text-encoder embeddings/linears too; Swift loads those through
the shared `MFluxEmbedding`/`MFluxLinear` group-quant paths.
`qwen-image-edit` q4 and q5 are live-proven for text-image edit after the VL-grid
conditioning fix. Source trace: mflux passes `vl_width/vl_height` into
`QwenEditUtil.create_image_conditioning_latents`, and that utility uses those
dimensions for source-image VAE conditioning when present. Swift now mirrors
that path in `QwenImageEditSupport.swift`: square source conditioning is
384x384 -> 24x24 -> 576 static tokens, not 1024x1024 -> 64x64 -> 4096 tokens.
q4 live proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-determinism-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`
has a coherent same-prompt deterministic repeat (blue edit SHA
`005ab8baddfe9b7a94aa83f8ddd22d192e7e5a0275c556dcf2ead76a565e474a` for turns
1 and 3) and a different coherent green-pear edit (SHA
`815711be73a9e89599b3e97f9f15196115875103f9407d7b1b61bab33de8e3b4`).
q5 live proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q5-determinism/Qwen-Image-Edit-mflux-q5-load.json`
has a coherent same-prompt deterministic repeat (blue edit SHA
`5cd5d9197bd659bd8b59b4a2f2bca413266146ad4e08249289d5fa6a8025fa4e` for turns
1 and 3) and a different coherent green-pear edit (SHA
`d2c6c4eb4a19dcf48122b5216fc15ac37b9f5aa49c15f596acd1276a4df57034`).
Shape proof:
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-conditioning-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`
(`latents_shape=1x576x64`) and
`docs/local/vmlx-flux-probes/2026-06-16-qwen-edit-q4-denoise-after-cond-fix/Qwen-Image-Edit-mflux-q4-load.json`
(`combined_velocity_shape=1x1600x64`). Do not expose q3 because its
text-encoder index references missing `text_encoder/3.safetensors`; keep q6
blocked until its local bundle is complete, and masks/inpaint are not wired yet.
Current Qwen edit rejects non-null `mask` before the pipeline loads; keep the UI
mask control hidden until real mask conditioning lands.

The sections below are the grounded port notes and transcription record. Older
"next" checkboxes may describe the sequence that produced the current native
txt2img implementation; the live status above is the source of truth.

> Do NOT mark this done without a live same-seed/different-prompt proof (the HARD
> RULE). The Z-Image proof in `OSAURUS_VMLX_FLUX_INTEGRATION_SPEC.md` §11 is the bar.

---

## Why Z-Image is the right template

Qwen-Image and Z-Image share the modern text-to-image recipe that
`vMLXFluxKit` + `ZImageNative.swift` already implement end-to-end:

- A **decoder-LM text encoder** (Z-Image: Qwen-style 2560-dim encoder; Qwen-Image:
  Qwen2.5-VL text tower) producing per-token hidden states as conditioning —
  `ZImageTextEncoder` is a near-drop-in pattern (embed_tokens → N× {RMSNorm,
  q/k/v/o_proj with q/k norm, RoPE attention, gated MLP}).
- An **MM-DiT** transformer that concatenates caption + image streams with
  RoPE position ids and adaLN-zero timestep modulation — `ZImageTransformer`
  (patchify + caption-concat + noise/context refiners + unified blocks +
  final layer + unpatchify) is the template.
- An **AutoencoderKL** VAE decoder — `ZImageVAEDecoder` (conv_in → mid_block →
  up_blocks → conv_out) is reusable; Qwen-Image's VAE has the same family shape.
- The **FlowMatch Euler scheduler**, **mflux 4-bit weight decode** (scale
  tensors), **tokenizer bridge** (`AutoTokenizer.from(modelFolder:)`), and
  **PNG IO** are all model-agnostic in `vMLXFluxKit` and already used by Z-Image.

So the port is mostly: (a) parse Qwen-Image's config, (b) map its checkpoint keys
to module properties, (c) match its exact text-encoder + DiT topology, (d) wire
image conditioning for `-edit`.

---

## Step-by-step

### 0. Stage weights + capture ground truth
- Copy `qwen-image-mflux-4bit` to `~/.mlxstudio/models/image/` (internal SSD).
- Run the **reference mflux (Python)** generation once with a fixed seed + prompt
  to get a ground-truth image + the intermediate shapes (text-encoder output dim,
  latent channels, patch size, VAE scale/shift). These are the oracle for the port.
- `vmlxflux-probe --model qwen-image-mflux-4bit --json` (scan only) to confirm the
  component layout (`transformer/`, `text_encoder/`, `vae/`, `tokenizer/`).

### 1. Config parse
- Read `transformer/config.json` + `text_encoder/config.json` + `vae/config.json`.
- Pull: hidden dim, num layers, num heads, head dim, patch size, in-channels,
  text hidden dim, RoPE theta, VAE scale/shift. Mirror the `ZImageNative` static
  constants block with Qwen-Image's numbers (do NOT reuse Z-Image's 3840/2560/30).

### 2. Weight key-map (the hard part)
- Enumerate `transformer/*.safetensors` keys (use `WeightLoader` which already
  merges component shards into `componentWeights["transformer"|"text_encoder"|"vae"]`).
- Build a `QwenImageWeightStore` modeled on `ZImageWeightStore` with the
  candidate-key fallbacks for Qwen-Image's naming (Qwen-Image uses
  `transformer_blocks.{i}.attn.*`, `img_mod`/`txt_mod`, `img_mlp`/`txt_mlp` —
  verify against the actual checkpoint, do not assume).
- For 4-bit mflux: reuse `ZImageWeightStore.linear(component:prefix:...)`'s
  scale-tensor dequant path (Qwen-Image mflux-4bit packs the same way).

### 3. Text encoder
- Qwen-Image uses a Qwen2.5-VL text tower. Start from `ZImageTextEncoder`; adjust
  dim/layers/heads, the prompt template (Qwen-Image has its own chat-style
  template — capture it from the reference), and the pooled vs per-token output
  contract. Feed per-token hidden states as `capFeats`.

### 4. Transformer (MM-DiT)
- Clone `ZImageTransformer`; match Qwen-Image's block structure (it has dual
  img/txt modulation per block — closer to Flux's double-stream than Z-Image's
  unified blocks, so cross-reference `FluxDiT.swift`'s `FluxDoubleStreamBlock`).
- 3-axis RoPE: confirm whether Qwen-Image uses Z-Image-style 1-grid RoPE or
  Flux-style 3-axis (time,H,W) — this is a known open TODO in `FluxDiT.swift`.

### 5. VAE
- Reuse `ZImageVAEDecoder`; verify channel progression + scale/shift against
  `vae/config.json` (Qwen-Image VAE scale/shift differ from Z-Image's 0.3611/0.1159).

### 6. `qwen-image-edit` image conditioning
- Implement `ImageEditor.edit(_:)`: VAE-**encode** the source image to a latent,
  blend with noise per `strength`, optionally apply the mask (white=edit), then
  run the same denoise loop. Needs the VAE **encoder** (the kit only has the
  decoder today — add `ZImageVAEEncoder`-style conv stack or reuse Qwen-Image's).

### 7. Live proof (gate — do not skip)
- `vmlxflux-probe --model qwen-image-mflux-4bit --generate --seed S --steps N
  --turn "A" --turn "B" --turn "A"` → turns 1≡3 byte-identical (determinism),
  turn 2 ≠ 1 (prompt-sensitivity), all coherent + prompt-accurate.
- For `-edit`: prove the output preserves source structure and applies the prompt
  edit (e.g. recolor) — compare against the masked region.
- Compare against the step-0 reference image for fidelity.

---

## Effort + risk

- **Biggest risk:** the weight key-map (step 2) and the exact text-encoder template
  (step 3) — wrong keys/topology produce coherent-looking but prompt-insensitive or
  garbled output. The same-seed/different-prompt + reference-image checks catch this.
- **Reuse ratio:** ~70% of the machinery (scheduler, VAE decoder pattern, weight
  loader, tokenizer, IO, mflux dequant, patchify/RoPE/adaLN scaffolding) is shared.
- **Shared payoff:** the T5-XXL encoder needed for `flux1-*`/`flux2-klein` is a
  separate port; Qwen-Image does NOT need T5 (it uses Qwen2.5-VL), so Qwen-Image is
  the better *next* target than the Flux family.

## GROUNDED mflux source structure (/tmp/mflux-ref/src/mflux/models/qwen, 2026-06-15)
Qwen is BIGGER than flux — 3 heavy components:
1. **Text encoder = Qwen2.5-VL** (`qwen_text_encoder/`): a full decoder LM
   (`qwen_text_encoder.py`, `qwen_encoder_layer.py`, `qwen_attention.py`, `qwen_mlp.py`,
   `qwen_rms_norm.py`, `qwen_rope.py`) PLUS a vision transformer
   (`qwen_vision_*.py`, patch_embed/patch_merger/vision_block/vision_attention/
   vision_rotary). For **txt2img** only the LM path is needed (`qwen_prompt_encoder.py`);
   for **EDIT** the vision tower encodes the input image too
   (`qwen_vision_language_prompt_encoder.py`). The LM is Qwen2.5-style (RMSNorm, GQA,
   RoPE, SwiGLU) — reuse the MFluxStore + the z-image Qwen-style attention pattern as a start.
2. **VAE = 3D causal VAE** (`qwen_vae/`): `qwen_image_decoder_3d`, `causal_conv_3d`,
   `res_block_3d`, `mid_block_3d`, `up_block_3d`, `attention_block_3d`, `resample_3d`,
   `rms_norm`. Like the Wan video VAE (vMLXFluxVideo/WanVAE3D has a CausalConv3d shim to reuse).
3. **Transformer = QwenTransformer** MMDiT (`qwen_transformer.py`) — double-stream like flux,
   reuse FluxJoint/Single block patterns adapted to qwen keys/dims.

Flow (txt2img `variants/txt2img/qwen_image.py`): text_encoder(prompt)→embeds; loop
transformer(t, latents, embeds)→scheduler; VAE.decode. Edit adds image→vision→embeds + img latents.

**Sequencing:** (1) qwen txt2img first (LM encoder only, simpler) → prove; (2) then edit
(add vision tower + image conditioning). Download: fcreait/Qwen-Image-Edit-mflux (staging).
Biggest new pieces: Qwen2.5 LM encoder + 3D VAE. Est. 3-5 iterations.

## GROUNDED checkpoint keys (carsenk/qwen-image-mflux-4bit, 24GB, staged, has tokenizer.json)
Components: text_encoder, transformer, vae, tokenizer.
- **text_encoder = Qwen2.5 LM (NOT quantized, bf16)** — keys: `encoder.embed_tokens.weight`,
  `encoder.layers.{0..27}.{input_layernorm, post_attention_layernorm}.weight`,
  `...self_attn.{q,k,v}_proj.{weight,bias}` (q/k/v HAVE bias), `...self_attn.o_proj.weight` (no bias),
  `...self_attn.rotary_emb.inv_freq`, `...mlp.{gate,up,down}_proj.weight`, `encoder.norm.weight`.
  28 layers. **This is the SAME arch as z-image's working ZImageTextEncoder** (Qwen2.5: RMSNorm,
  GQA attn with q/k/v bias, RoPE from inv_freq, SwiGLU). REUSE that pattern — biggest shortcut.
  Output: per-token hidden states = the transformer conditioning. (Need to find hidden dims/head
  counts from config.json + the qwen-image prompt template/system prompt from mflux qwen_prompt_encoder.)
- **transformer = MMDiT (quantized 4bit, weight/scales/biases)** — `img_in`, `time_text_embed.
  timestep_embedder.linear_{1,2}`, `transformer_blocks.N.*` (double-stream, like flux —
  reuse FluxJoint patterns adapted to qwen keys), `norm_out.linear`, `proj_out`. No context_embedder
  (text enters differently — check qwen_transformer.py). Use MFluxStore.
- **vae = 3D causal VAE (NOT quantized)** — `decoder.conv_in.conv3d`, `decoder.mid_block.{resnets.N.
  {conv1,conv2}.conv3d, norm1/norm2, attentions.N.{norm, to_qkv, proj}}`, `decoder.up_block{0..N}.
  resnets.N.conv{1,2}.conv3d` (+ upsamplers), `decoder.norm_out`, `decoder.conv_out.conv3d`. conv3d
  (causal) + RMSNorm. REUSE vMLXFluxVideo/WanVAE3D's CausalConv3d shim (collapse T into batch → Conv2d).
  Qwen VAE: latent_channels 16, temporal compression. For txt2img the temporal dim = 1.

## NEXT (this is the port-write sequence)
1. Read config.json (text_encoder hidden/heads/kv_heads/head_dim/layers, transformer dims, vae scale/shift).
2. Read mflux qwen_image.py (txt2img flow) + qwen_prompt_encoder.py (prompt template) +
   qwen_transformer.py (block structure + how text conditioning enters) + qwen_vae.py (decode + scale/shift).
3. Write Common/QwenImageNative.swift: QwenTextEncoder (reuse z-image Qwen pattern on MFluxStore),
   QwenTransformer (MMDiT), Qwen3DVAEDecoder (CausalConv3d). Pipeline + wire QwenImage.generate.
4. De-risk stderr finite stats → live-prove coherent. Then qwen-image-edit (add vision tower).

## GROUNDED qwen text-encoder dims + gotchas (mflux source, 2026-06-15)
**QwenTextEncoder = Qwen2.5-VL LM (txt2img uses LM only, no vision tower):**
- hidden 3584, 28 layers, 28 attn heads, 4 KV heads (GQA, 7 groups), head_dim 128,
  intermediate 18944, RMSNorm eps 1e-6, rope_theta 1e6, SwiGLU MLP (gate/up/down).
- Attn: q_proj(3584→3584,bias), k_proj(3584→512,bias), v_proj(3584→512,bias),
  o_proj(3584→3584,NO bias). Keys: `encoder.layers.N.self_attn.{q,k,v}_proj.{weight,bias}`,
  `o_proj.weight`, `input_layernorm`/`post_attention_layernorm`.weight, `mlp.{gate,up,down}_proj.weight`,
  `encoder.embed_tokens.weight`, `encoder.norm.weight`. NOT quantized (bf16).
- **mRoPE** (rope_scaling mrope_section [16,24,24]) — Qwen2.5-VL multimodal RoPE; for pure TEXT
  all 3 position sections use the text position (so effectively standard RoPE over head_dim with
  theta 1e6 split [16,24,24]*2=128). Verify against qwen_rope.py.
- **GOTCHA drop_idx=34**: output drops the FIRST 34 tokens (the qwen-image system-prompt template)
  from hidden states → those become prompt_embeds. So tokenization MUST use qwen-image's chat template
  (a ~34-token system prefix). Get template from mflux qwen_vision_language_tokenizer / tokenizer_config.
  Also produces an attention MASK (valid-token mask) used by the transformer.
- **CFG**: qwen-image uses guidance (default 4.0) — TWO transformer passes/step (pos+neg prompt),
  guided = neg + guidance*(pos-neg). negative_prompt default " ". More steps than turbo (read default).

**Transformer** = MMDiT (quantized): `img_in`, `time_text_embed.timestep_embedder.linear_{1,2}`,
`transformer_blocks.N.*`, `norm_out.linear`, `proj_out`. Text enters as encoder_hidden_states+mask
(read qwen_transformer.py for block structure + how mask/text used — likely double-stream like flux).
**VAE** = 3D causal conv (conv3d), txt2img temporal dim=1; reuse WanVAE3D CausalConv3d.

PROGRESS:
- [x] `Common/QwenImageNative.swift` QwenTextEncoder COMPILES — Qwen2.5 LM (embed_tokens, 28×QwenLMLayer
  {RMSNorm→GQA attn(q/k/v bias, o no bias, repeat_kv 7×, standard RoPE theta1e6, causal mask)→ +res,
  RMSNorm→SwiGLU→ +res}, encoder.norm, drop first 34 tokens). On MFluxStore (text_encoder bf16, not quant).
- [ ] tokenizer wrapper with qwen-image CHAT TEMPLATE (the ~34-token system prefix that drop_idx=34 removes
  — get from mflux qwen_vision_language_tokenizer.py / qwen_image_processor; tokenizer_config chat_template empty).
- [ ] QwenTransformer (MMDiT — read qwen_transformer/qwen_transformer.py: img_in, time_text_embed, blocks,
  norm_out, proj_out, how encoder_hidden_states+mask enter; reuse FluxJoint patterns on MFluxStore quant).
- [ ] Qwen3DVAEDecoder (conv3d causal — read qwen_vae/qwen_vae.py for scale/shift + structure; reuse WanVAE3D
  CausalConv3d shim; txt2img temporal=1).
- [ ] pipeline + CFG (guidance 4.0, pos+neg, 2 transformer passes/step) + wire QwenImage.generate.
- [ ] de-risk stderr finite, live-prove. Default steps/guidance from ModelConfig.qwen_image.
GROUNDED transformer (qwen_transformer.py): MMDiT, in_ch 64, out_ch 16, **60 layers**, 24 heads × 128 =
inner 3072, joint_attention_dim 3584 (text), patch 2. img_in Linear(64→3072), txt_norm RMSNorm(3584),
txt_in Linear(3584→3072), time_text_embed = QwenTimeTextEmbed (timestep-ONLY: QwenTimesteps sinusoidal
proj256 scale1000 → QwenTimestepEmbedding 256→3072; NO pooled text). pos_embed = QwenEmbedRopeMLX(theta
10000, axes_dim [16,56,56], scale_rope=True) → separate img_rotary + txt_rotary. norm_out =
AdaLayerNormContinuous(3072). proj_out Linear(3072→2*2*16=64).
QwenTransformerBlock (×60): img_mod_linear/txt_mod_linear (3072→6*3072), split into mod1,mod2 (each 3*3072
= shift,scale,gate). img_norm1/2, txt_norm1/2 = LayerNorm affine=false. _modulate(x,mod)=x*(1+scale)+shift,
returns (mod_x, gate). Flow: modulate img+txt w/ mod1 → QwenAttention(img,txt,mask,rope) → gate1 residual
on both; modulate w/ mod2 → img_ff/txt_ff (QwenFeedForward) → gate2 residual. Returns (encoder, hidden).
STILL TO READ next iter: qwen_attention.py (joint attn: how img+txt concat, rope applied, mask), qwen_feed_forward.py,
qwen_rope.py QwenEmbedRopeMLX (3-axis scale_rope img/txt split), qwen_vae.py (+3d blocks, scale/shift),
chat template (qwen_vision_language_tokenizer), ModelConfig.qwen_image (default steps≈? guidance 4.0).
Then write QwenTransformer + Qwen3DVAEDecoder + tokenizer + pipeline+CFG into QwenImageNative.swift, wire
QwenImage.generate, de-risk finite, live-prove. (60-layer transformer + 3D VAE = big; ~2-3 more iterations.)

GROUNDED (all transformer internals — ready to transcribe):
- **QwenAttention** (joint): to_q/k/v + add_q/k/v_proj (all Linear dim→dim, bias), norm_q/k/added_q/added_k
  = RMSNorm(head_dim=128, eps1e-6). Reshape q/k/v to (b, seq, heads24, 128) — KEEP (b,s,h,d) for rope.
  Apply RMSNorm on q/k (img: norm_q/norm_k; txt: norm_added_q/norm_added_k). Apply RoPE (complex-pair, see
  below) to img q/k with img_cos/sin, txt q/k with txt_cos/sin. concat [txt, img] on axis=1 (seq).
  _compute_attention: transpose to (b,h,s,d), SDPA scale 1/sqrt(128), mask (additive from text mask),
  transpose back, reshape (b,s,3072). Split [: seq_txt]=txt, [seq_txt:]=img. img→attn_to_out[0], txt→to_add_out.
  RoPE apply (complex): x→(...,−1,2) real/imag; cos/sin (seq,64)→[None,:,None,:]; out_real=r*cos−i*sin,
  out_imag=r*sin+i*cos; stack→reshape back.
- **QwenFeedForward**: mlp_in(dim→4*dim,bias)→gelu_approx→mlp_out(4*dim→dim,bias).
- **QwenTimeTextEmbed**: QwenTimesteps(sinusoidal half128, max_period 1e4, ×scale 1000, flip sin/cos halves)
  → (1,256); QwenTimestepEmbedding linear_1(256→3072,bias)→silu→linear_2(3072→3072,bias). = text_embeddings (timestep-only).
- **QwenEmbedRopeMLX** (theta 1e4, axes_dim [16,56,56], scale_rope=true): precompute pos_freqs/neg_freqs
  (np): for each axis dim, scales=arange(0,dim,2)/dim, omega=1/theta^scales, freqs=outer(index,omega),
  stack[cos,sin]. pos_index=0..4095, neg_index reversed negative. _compute_video_freqs(frame=1,h//16,w//16):
  axes_splits=[8,28,28]; split pos/neg freqs by cumsum; frame axis from pos[0][idx:idx+frame] bc to (f,h,w);
  height axis: scale_rope→concat[neg[1][-(h-h//2):], pos[1][:h//2]] bc; width axis similarly; concat on axis −2
  → (seq, 64, 2). img_cos/sin = (seq,64). txt_cos/sin = pos_freqs[max_vid_index : +max_len] (max_vid_index =
  max(h//2,w//2)). Implement in Swift by precomputing the freq tables.
- **norm_out** = AdaLayerNormContinuous (REUSE flux FluxAdaNormContinuous — scale then shift, no-bias linear 3072→6144).
- **Pipeline**: latents packed like flux (1,(h//16)(w//16),64? — qwen in_channels 64, check QwenLatentCreator pack);
  loop: noise_pos=transformer(t,latents,prompt_embeds,mask); noise_neg=transformer(t,latents,neg_embeds,neg_mask);
  guided = neg + guidance*(pos−neg); FlowMatch step; unpack; VAE decode. CFG guidance 4.0.
GROUNDED VAE (qwen 3D causal-conv decoder, 106 decoder keys, conv keys end `.conv3d.{weight,bias}`):
- **De-norm is PER-CHANNEL** (not scalar): latents(1,16,1,h/8,w/8) * LATENTS_STD[16] + LATENTS_MEAN[16]
  (values in qwen_vae.py — hardcode the two 16-vectors). Then post_quant_conv (CausalConv3D 16→16 k1) → decoder.
  Latent packing = SAME as flux (unpack → (1,16,h/8,w/8) → reshape to 5D (1,16,1,h/8,w/8)).
- **QwenImageDecoder3D**: conv_in CausalConv3D(16→384,k3); mid_block MidBlock3D(384, 1 layer = 2 resnets +
  1 attention); up_block0 UpBlock3D(384→384, 2 res, upsample3d); up_block1(192→384?, 2 res, upsample3d, has
  skip_conv); up_block2(192→192, 2 res, upsample2d); up_block3(96→96, 2 res, no upsample); norm_out RMSNorm(96);
  conv_out CausalConv3D(96→3,k3); silu before conv_out. Channel flow 384→384→192→192→96→96→3.
  Keys: decoder.{conv_in.conv3d, mid_block.{resnets.{0,1}.{conv1,conv2}.conv3d + norm1/norm2, attentions.0.
  {norm, to_qkv, proj}}, up_block{0..3}.{resnets.{0,1,2}.{conv1,conv2}.conv3d + norm1/norm2 (+skip_conv.conv3d
  on channel-change blocks), upsamplers.0.{resample_conv, time_conv.conv3d}}, norm_out, conv_out.conv3d}.
  VAE is NOT quantized (weight/bias only). RMSNorm here = QwenImageRMSNorm (spatial, images=False).
- For txt2img temporal dim=1: CausalConv3D can reuse vMLXFluxVideo WanVAE3D's CausalConv3d shim (collapse T
  into batch → Conv2d) since T=1. upsample3d temporal upsample is a no-op at T=1 (just spatial 2x); upsample2d
  = spatial 2x; the time_conv/resample_conv handle it.
NEXT: read /tmp/mflux-ref/.../qwen_vae/{qwen_image_causal_conv_3d, qwen_image_res_block_3d, qwen_image_up_block_3d,
qwen_image_attention_block_3d, qwen_image_mid_block_3d, qwen_image_resample_3d, qwen_image_rms_norm}.py + compare
to vMLXFluxVideo/WanVAE3D.swift (reuse its 3D blocks). Also chat template (qwen tokenizer applies a system prompt —
the drop_idx=34; get exact template from qwen_vision_language_tokenizer.py / qwen_image_processor) + qwen-image
default steps (ModelConfig qwen-image line 484+). Write Qwen3DVAEDecoder + tokenizer + QwenImagePipeline (CFG
guid4.0 pos+neg, FlowMatch) + wire QwenImage.generate. Compile, de-risk finite, live-prove.

## Reference files
- Template: `Libraries/vMLXFluxModels/ZImage/ZImageNative.swift` (proven).
- Flux double-stream block (for MM-DiT): `Libraries/vMLXFluxKit/FluxDiT.swift`.
- Shared kit: `FlowMatchScheduler`, `VAE`, `WeightLoader`, `MathOps`, `LatentSpace`.
- API contract + proof bar: `docs/OSAURUS_VMLX_FLUX_INTEGRATION_SPEC.md`.
