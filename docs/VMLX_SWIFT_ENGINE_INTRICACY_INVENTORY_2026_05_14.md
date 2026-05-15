# vMLX Swift Engine Intricacy Inventory

Snapshot date: 2026-05-14.

This is a source-led inventory of the current consolidated `vmlx-swift`
runtime. It is not a production-readiness claim. A row is production-ready only
after live multi-turn output proves coherence, token/s, stop behavior, and the
correct cache topology for the model family. Config, template, and unit-test
coverage prove wiring, not user-facing model quality.

## Runtime Ownership Boundary

`vmlx-swift` owns:

- Swift package graph, vendored Jinja/Tokenizers/Hub/Generation code, MLX/Cmlx
  integration, and model factories.
- Model load, tokenizer resolution, JANG/JANGTQ metadata merge, sidecar sniffing,
  weight sanitization, and `generation_config.json` defaults.
- Text/VL/Omni input preparation, chat templates, tool schemas, reasoning parser
  routing, stop strings, and streaming event routing.
- `GenerateParameters`, sampling, penalties, KV policy, speculative strategy
  selection, acceleration selection, and compile flags.
- `TokenIterator`, `Evaluate.generate`, `ModelContainer`, `BatchEngine`, and the
  cache stack.
- Prefix cache, paged cache, disk L2, SSM companion state, ZAYA CCA state,
  DSV4 hybrid-pool cache, media/cache-scope salts, and TurboQuant KV.
- JANG/JANGTQ routed expert dispatch, TurboQuant codebook kernels, JangPress /
  MLXPress routed-weight residency, and active-expert diagnostics.
- VLM media processors and Nemotron Omni RADIO/Parakeet media encoders.
- Distributed planning/tools at package level.

Osaurus still owns Electron UI, tray/buttons/i18n, HTTP `/v1` and `/admin`
routes, packaged app launch, deep sleep/wake process policy, user settings, and
server lifecycle. Those become downstream gates after Osaurus repins/migrates to
this package.

## Top-Level Request Path

1. `ResolvedModelConfiguration` points at a local bundle and tokenizer source.
2. `LLMModelFactory` or `VLMModelFactory` reads `config.json`.
3. JANG bundles merge selected `jang_config.json` fields into config data before
   model dispatch.
4. Factories select a model class by `model_type`, `weight_format`, sidecar
   presence, nested `text_config`, and multimodal config.
5. `JangLoader` resolves tokenizer fallback, tokenizer-class substitution, and
   chat-template sidecar substitution.
6. `loadWeights` sanitizes and loads MLX weights, with JANG per-layer quant
   inference when needed.
7. `ModelConfiguration` records EOS IDs, extra EOT tokens, generation defaults,
   tool-call format, and reasoning parser stamp.
8. `ModelContainer` wraps the `ModelContext`, optional cache coordinator, and
   optional JangPress/MLXPress runtime.
9. Callers either use direct `ModelContainer.generate`, `TokenIterator`, or
   `BatchEngine.generate/submit`.
10. Streaming output routes through detokenization, reasoning parsing, tool-call
    extraction, stop-string matching, and final `GenerateCompletionInfo`.

## Generation Parameters

`GenerateParameters` is the main runtime control surface:

- `prefillStepSize`: chunk size for prompt processing.
- `maxTokens`: generated token cap.
- `maxKVSize`: rotating-window cap for KV cache.
- `kvBits`, `kvGroupSize`, `quantizedKVStart`: legacy affine KV quantization.
- `kvMode`: `.none`, `.affine(bits, groupSize)`, or `.turboQuant(keyBits,
  valueBits)`.
- `enableCompiledDecode`, `compiledMaxCacheLength`: single-sequence compile
  path controls.
- `enableCompiledBatchDecode`, `compiledBatchBuckets`: batch compile controls.
- `accelerationMode`: `metal`, `auto`, or `ane-coreml`; text decode fails closed
  for ANE unless a validated Core ML island exists.
- `temperature`, `topP`, `topK`, `minP`: sampler controls.
- `repetitionPenalty`, `presencePenalty`, `frequencyPenalty` and context sizes.
- `draftStrategy`: optional autoregressive/diffusion speculative path; plain
  autoregressive remains default.
- `extraStopStrings`: decoded visible-output stop strings.
- Reasoning parsing is routing only. The engine must not bias or force
  `</think>` tokens to make a reasoning block close; if a model never emits
  its family close marker, the row is a runtime/template/cache failure to
  investigate, not something to hide with decode logits.

Sampling behavior:

- `temperature == 0` uses argmax.
- Sampling filters run in Python mlx-lm order: top-p, then min-p, then top-k.
- `repetitionPenalty == nil`, `0`, or `1.0` means no repetition processor.
- Bundle defaults come from `generation_config.json` only when callers ask
  `ModelContainer.defaultGenerateParameters(...)` or equivalent; direct
  `GenerateParameters` remains explicit.

## Scheduler Modes

`BatchEngine` has two distinct execution modes:

- B=1 solo fast path: uses `TokenIterator`-style generation, with cache
  coordinator support and optional compiled single-slot promotion.
- B>1 continuous batching: queues requests, admits up to `maxBatchSize`, runs
  prefill per slot, then batches decode tokens through a shared forward pass.

Scheduler nuances:

- Initial coalescing window exists only for B>1 engines.
- Prefill is sequential per slot because heterogeneous prompt lengths waste
  compute if padded.
- Decode uses direct slot caches for B=1 and transient batch wrappers for B>1.
- Stream termination cancels orphaned slots so Metal work does not continue
  after the client disconnects.
- Cancellation finishes pending/active slots with `.cancelled`.
- Memory cache purges happen periodically and after long-context completions.
- DSV4 `HybridPoolCache` slots are serialized against other active slots because
  its compressor/indexer pools are not representable by ordinary batch wrappers.
- DSV4 prefill window is raised to the full prompt for hybrid-pool cache slots.

## Cache Taxonomy

Per-layer cache types that matter:

- `KVCacheSimple`: standard full KV cache.
- `RotatingKVCache`: sliding-window/ring-buffer KV with keep/max/step/offset
  metadata.
- `QuantizedKVCache`: affine quantized KV.
- `TurboQuantKVCache`: TurboQuant compressed KV cache.
- `MambaCache`: two-slot recurrent SSM state.
- `ArraysCache`: linear-attention/recurrent state cache used by Bailing/Ling and
  related families.
- `CacheList`: composite per-layer cache such as Mamba plus KV, or rotating plus
  Mamba.
- `ZayaCCACache`: KV plus path-dependent `conv_state` and `prev_hs`.
- `HybridPoolCache`: DSV4 rotating window plus CSA/HSA/compressor/indexer pool
  state and incomplete-window buffers.

Compile classification:

- `.simple`: `KVCacheSimple` / `CompilableKVCache`; compile eligible.
- `.turboQuant`: `TurboQuantKVCache` / `CompilableTurboQuantKVCache`; compile
  eligible after the RoPE offset fix.
- `.rotating`: `RotatingKVCache` / `CompilableRotatingKVCache`; compile
  eligible for pure rotating families.
- `.cacheList`: composite cache; compile eligible only when every sub-cache is
  compile-ready.
- `.mamba`: pure Mamba/Arrays; not compile-ready as a production path.
- `.zayaCCA`: ZAYA CCA plus no-op simple slots; not compile-ready yet.
- `.heterogeneous`: mixed cache families; honest uncompiled fallback.

## Cache Coordinator

`CacheCoordinatorConfig` controls:

- `usePagedCache`: in-memory paged prefix cache.
- `enableDiskCache`: disk L2 safetensors/SQLite cache.
- `pagedBlockSize`: token block size; lowered to 16 for hybrid models in
  `ModelContainer.enableCachingAsync`.
- `maxCacheBlocks`, `diskCacheMaxGB`, `diskCacheDir`.
- `ssmMaxEntries`, `enableSSMReDerive`.
- `modelKey`: cross-model cache poisoning guard, also scoped by MoE top-k
  overrides.
- `defaultKVMode`, `defaultMaxKVSize`, `longPromptMultiplier`: coordinator-owned
  KV sizing defaults.

Fetch order:

1. Paged cache, unless the model is paged-incompatible.
2. Disk L2, probing exact and prompt-boundary lengths.
3. SSM companion resolution when the model is hybrid.
4. Miss.

Critical guards:

- Hybrid paged hits require SSM companion state unless a format-v2 disk payload
  already carries path-dependent state.
- Hybrid exact full hits are avoided or rolled back when re-feeding the last
  token would double-count recurrent state.
- Media partial hits that leave media placeholders in the suffix roll back to
  full prefill.
- DSV4 and other disk-backed topologies set `isPagedIncompatible=true` so paged
  token matches do not suppress the disk serializer that actually understands
  their state.
- Disk-restore arrays are eagerly evaluated before prefill to avoid lazy restore
  work being fused into the same Metal command buffer as model kernels.
- Cache trim after full disk hits is eagerly evaluated before the seed forward.

Store behavior:

- Stores prompt-boundary snapshots after prefill.
- Stores post-answer boundaries only when normal stop and cache coverage prove
  the generated tokens are present.
- Paged cache stores block-aligned per-layer KV where representable.
- Disk cache stores raw per-cache layer payloads via `TQDiskSerializer` v2.
- Hybrid SSM state either re-derives from prompt boundary or extracts from the
  prompt snapshot depending on topology.

## Disk L2 Serializer

`TQDiskSerializer` format v2 layer kinds:

- `.tq`: compressed `TurboQuantKVCache`.
- `.kv`: standard KV pairs or fill-phase TQ KV.
- `.mamba`: Mamba conv/hidden state plus offset.
- `.qkv`: affine quantized KV tensors and metadata.
- `.rotating`: ring buffer plus keep/max/step/offset/index metadata.
- `.deepseekV4`: DSV4 rotating window plus compressor/indexer/pool state.
- `.cacheList`: tagged sub-cache payloads.
- `.zayaCCA`: ZAYA KV plus CCA `conv_state` and `prev_hs`.
- `.skip`: known unsupported or empty cache layer.

Restore is intentionally conservative: incompatible shape, wrong bit/group
metadata, missing required state, or unrecognized layer kind must fall back to
fresh prefill rather than produce a false hit.

## Media And Cache Identity

`UserInput` supports:

- `Prompt.text`, raw message dictionaries, and structured `Chat.Message`.
- Images from `CIImage`, URL, or `MLXArray`.
- Video from URL, AVAsset, or decoded frames depending on processor support.
- Audio from URL, PCM samples, MLXArray, or pre-encoded embedding.
- `tools` and `additionalContext`.

`LMInput` carries processed text plus optional processed image/video/audio,
media token IDs, and cache-scope salt.

Cache salt pieces:

- `computeMediaSalt`: hashes image/video/audio raw bytes, shape, dtype, and audio
  sample rate.
- `cacheScopeSalt`: hashes reasoning-affecting context such as
  `enable_thinking` and `reasoning_effort`.
- `computeCacheSalt(input, parameters)`: also folds KV cache policy, max-KV
  policy, and serializer contract.

Important VLM rules:

- Text-only turn after a media turn may have nil media salt, but the prefix
  resume must not split a media-token region.
- Same text plus different image/video/audio must miss.
- Same text plus same media plus different reasoning mode must miss.
- Same prompt plus different KV policy must miss.
- VL processors must emit correct placeholder counts and position vectors.
- Qwen VL-style models need `[3, batch, seq]` position IDs, image/video THW
  grids, and MRoPE delta vectors.
- Extent guards must reject non-finite or non-positive dimensions before shape
  math reaches MLX kernels.

## Nemotron Omni: RADIO, Parakeet, Live Voice

Nemotron Omni combines:

- `NemotronHModel` text backbone.
- `NemotronHRADIOVisionModel` for images/video.
- `NemotronHVisionMLPProjector` (`mlp1`) from RADIO features to LLM hidden.
- `NemotronHParakeetEncoder` for audio.
- `NemotronHSoundProjector` from Parakeet frames to LLM hidden.

RADIO details:

- Image pixels are CLIP-normalized, passed through RADIO, stripped of
  class/register tokens, reshaped to square patch grids, pixel-shuffled by
  downsample ratio, then projected with `mlp1`.
- Video uses RADIO's `video_embedder` with channel-stacked temporal patches.
- Image and video both splice into the same image context token ID; ordering is
  image first, video second.

Parakeet details:

- File input uses AVFoundation decode/resample to 16 kHz mono Float32.
- In-memory audio is linearly resampled to 16 kHz if needed.
- Mel extraction uses Parakeet STFT/mel settings.
- Parakeet subsamples by factor 8, then sound projection maps to LLM hidden.
- `UserInput.Audio.preEncoded` preserves caller-supplied embeddings for low
  latency live voice; the processor still retains PCM for cache salt and final
  logical input identity.
- Multiple audio inputs are concatenated in prompt order.
- Prompt placeholders use `<sound>` plus repeated `<so_embedding>`.
- There is no neural audio-out decoder in this bundle; voice output is system
  TTS or external TTS, not model-native waveform generation.

Live voice helpers:

- `NemotronHOmniLiveAudioBuffer` keeps the whole turn while allowing incremental
  `consumeAvailableSamples()` polling.
- `NemotronHOmniMicRecorder` uses AVAudioEngine tap, converts to 16 kHz mono,
  and exposes retained snapshots.
- Concatenating independently encoded Parakeet chunks is unsafe as an audio
  equivalence claim unless embeddings and prompt placeholders are managed as one
  coherent turn.

## JANG, JANGTQ, And Routed Expert Residency

JANG/JANGTQ load nuances:

- `jang_config.json` is merged into runtime config for `weight_format`,
  `mxtq_bits`, `mxtq_seed`, routed expert bits, parser stamps, and tokenizer
  source hints.
- `mxtq_bits` may be flat, per-role, or nested per projection.
- JANGTQ_K can use different gate/up and down bit widths; fused gate/up requires
  matching gate/up bits.
- `jangtq_runtime.safetensors` sidecar codebook sniff is authoritative when
  bundle metadata is mislabeled.
- Nested `text_config` must receive mirrored JANGTQ fields for wrapped VL/text
  configs.

JANGTQ execution surfaces:

- `TurboQuantSwitchGLU` for routed MoE expert codebook matmul.
- `StreamingTurboQuantSwitchGLU` and `JANGTQStreamingExperts` for active-slice
  diagnostics.
- `JANGTQDenseLinear` for dense JANGTQ paths such as Mistral3 text/VLM.
- `JANGTQKernels` for Hadamard, codebook, gather, and fused gate/up operations.
- `JANGTQRuntimeCache` for sidecar codebook/tensor lookup.
- Hadamard path must preserve rank-2, rank-3, and rank-4 shapes.
- MiniMax's 1536 intermediate path is the canonical non-power-of-two
  `1024 + 512` split.

JangPress/MLXPress residency:

- Normal target is compression-first mmap residency: routed weights logically
  loaded, inactive expert pages advised cold, macOS reclaim/compression doing
  memory pressure relief.
- Permanent prestacked overlays are diagnostic opt-in only.
- Active-streaming slice reads are fallback diagnostics, not production proof.
- Low Activity Monitor `phys_footprint` plus usable token/s plus coherent
  multi-turn output are required together.
- Fast looping or hidden length-stop success is failure, not a pass.

## DSV4 Flash Special Topology

DSV4 is not DSV3/Kimi with a renamed model type. Its runtime has:

- mHC residual stream.
- CSA/HSA/HCA-style hybrid attention branches.
- Sliding-window attention.
- Compressor/indexer pool state.
- Branch incomplete-window buffers.
- Hash routing on early layers.
- sqrtsoftplus/gated MoE variants.
- JANGTQ routed expert variants and affine JANG variants.

DSV4-specific controls:

- `dispatchDeepseekV4` selects affine vs JANGTQ by `weight_format`, force env,
  sidecar-corrected metadata, and routed bit plan.
- `DSV4_JANGTQ_BITS` can override bit width for diagnostics.
- DSV4 EOS set includes role-boundary tokens that public configs can omit.
- `HybridPoolCache` sets paged-incompatible restore.
- Disk serializer kind `.deepseekV4` must preserve rotating window plus pool
  state.
- BatchEngine serializes DSV4 hybrid-pool slots and uses full-prompt prefill
  window for this topology.

Open DSV4 proof requirements:

- Long-context no-loop gate.
- Vector drift gate.
- CSA/HSA/SWA pool restore under multi-turn.
- Finalizer budget and role-token stop behavior.
- Speed matrix and reasoning matrix across JANGTQ variants.

## Hybrid SSM And Async Re-Derive

Path-dependent non-KV state exists in:

- Qwen3.5/Qwen3.6 GatedDelta/Mamba-style layers.
- Bailing/Ling linear attention and ArraysCache paths.
- Nemotron H hybrid Mamba/attention patterns.
- Jamba/Falcon-style composite caches.
- ZAYA CCA (`conv_state`, `prev_hs`).

Core functions:

- `cacheContainsPathDependentState`: detects recurrent/CCA state.
- `extractSSMStates` / `restoreSSMStates`: companion-state capture/restore.
- `SSMStateCache`: in-memory companion cache.
- `SSMCompanionDiskStore`: persistent companion store.
- `SSMReDerive`: clean prompt-boundary re-derive and prompt-boundary storage.
- Inline capture after prefill can avoid an extra re-derive pass when live cache
  already represents the prompt boundary.

Correctness rules:

- KV without matching path-dependent state is a false hit.
- Exact full disk hits for hybrid state can double-count the last token and must
  be rolled back or avoided.
- ZAYA CCA v2 disk payload already carries path-dependent state; it should not
  require a separate SSM companion entry.
- Media-bearing hybrid turns cannot re-derive state from text-only tokens.

## Reasoning, Jinja, And Tools

Template stack:

- Vendored Swift Jinja renders chat templates.
- `DefaultMessageGenerator` emits model-compatible message dictionaries.
- `additionalContext` carries `enable_thinking`, `thinking`,
  `reasoning_effort`, and model-specific kwargs.
- `VLMDefaultContextUserInputProcessor` seeds `enable_thinking=false` only when
  bundle capabilities explicitly say thinking is unsupported.
- DSV4 reasoning policy normalizes public `reasoning_effort` values and must
  not silently downgrade `max`.
- Kimi templates require `tojson(separators=(',', ':'))` compatibility and
  `enable_thinking` to `thinking` aliasing only when `thinking` is absent.
- ZAYA-VL sidecar templates must preserve vision placeholders while adding
  ZAYA XML tools in every loader-visible template source.

Reasoning routing:

- `ReasoningParser` splits streaming output into `.reasoning` and `.chunk`.
- `ReasoningParser.forPrompt` inspects the decoded prompt tail so thinking-off
  prompts do not start inside reasoning.
- Think-XML families include Qwen, DeepSeek, GLM, MiniMax, Kimi, Nemotron,
  Ling/Bailing, Laguna, ZAYA, and Hy3 aliases.
- Harmony families include Gemma4 and GPT-OSS.
- Unknown/non-reasoning families default to no parser, not think-XML.
- End-of-stream while still inside reasoning is surfaced via
  `unclosedReasoning`.

Tool formats:

- `json`
- `lfm2`
- `xml_function`
- `glm4`
- `gemma`
- `gemma4`
- `kimi_k2`
- `minimax_m2`
- `mistral`
- `llama3`
- `dsml`
- `zaya_xml`
- `hunyuan`

Tool parser selection priority:

1. Explicit caller configuration.
2. `jang_config.chat.tool_calling.parser`.
3. `jang_config.capabilities.tool_parser`.
4. `model_type` heuristic.

Tool parsing happens after reasoning parsing. That means tool calls emitted
inside a reasoning stream are routed according to channel and must not leak raw
JSON/XML into visible content.

## Family-Specific Notes

DeepSeek V4 Flash:

- DSV4 model classes, DSV4 reasoning policy, DSML tools, DSV4 EOS widening,
  JANGTQ bit resolution, and DSV4 disk serializer are wired.
- Live proof exists for short coherent DSV4 JANGTQ-K chat and targeted cache
  topology tests.
- Long-context/vector/speed matrix remains open.

DeepSeek V3 / Kimi K2.x:

- Shared DeepSeekV3/Kimi family dispatch.
- JANGTQ routes to `DeepseekV3JANGTQModel`.
- Kimi uses Kimi K2 tool parser and thinking alias rules.
- Kimi routed-weight residency remains its own memory/speed axis because routed
  expert reuse can be extremely low.

Qwen3.5 / Qwen3.6:

- Qwen35 and Qwen35MoE/JANGTQ dispatch exist.
- Hybrid SSM/linear state must use companion state or re-derive.
- Qwen VL requires MRoPE grids/deltas and media salt.
- Thinking-on can trap visible answer in reasoning under current live probes and
  needs matrix proof.

MiniMax M2.7:

- Standard and JANGTQ dispatch exist.
- MiniMax tool parser and think-XML parser are wired.
- JANGTQ_K gate/up/down bit split is represented.
- Compiled decode is currently denied for MiniMax until parity is proven.
- Low-footprint fast production path remains unproven.

ZAYA text:

- `ZayaModel`, `ZayaCCACache`, `BatchZayaCCACache`, and disk `.zayaCCA` are
  wired.
- JANGTQ gate/up and down bit widths are preserved.
- Salted growing-cache rows are the key correctness proof; generic semantic
  recall prompts are weaker and should not be used alone.

ZAYA VL:

- Processor and model are wired.
- Vision placeholders plus ZAYA XML tool sidecar shim are fixed at template
  level.
- Live VL generation/cache needs rerun after template/schema fixes.

Ling / Bailing:

- Bailing hybrid/MoE models are registered.
- ArraysCache/linear-attention recurrent state must be treated as path-dependent.
- Ling needs live multi-turn, SSM cache, and loop/coherence proof.

Gemma 4:

- Text and VLM paths are registered.
- Gemma4 harmony reasoning and Gemma4 tools are routed.
- Gemma4 mixed simple plus rotating cache is heterogeneous and stays uncompiled.
- Audio input is guarded because Gemma4 audio weights are not a wired audio path.

Nemotron Omni:

- Text, image, video, audio, RADIO, Parakeet, projectors, live PCM helpers, and
  pre-encoded audio are wired.
- No neural audio-out decoder exists in package.
- Full current live proof needs to be rerun in this consolidated branch before
  production claims.

Mistral / Laguna:

- Mistral3/Ministral wrappers route text/VLM correctly.
- Mistral4 text/VLM are registered.
- Laguna handles mixed quantization: dense affine quant plus routed JANGTQ
  codebook experts, or MXFP4 affine routed experts.
- Laguna tool parser maps to GLM4-style tools and think-XML reasoning.

Hy3 / Hunyuan:

- Registered under `hy_v3`, `hy3`, and `hy-v3`.
- Hunyuan parser and think-XML reasoning aliases are wired.
- MTP weights may exist, but base decode must stay plain autoregressive unless
  a family-specific MTP path proves parity.

GPT-OSS:

- Registered text model.
- Harmony reasoning stamp prevents raw channel markers from leaking.

LFM2 / LFM2-VL:

- Text and VL registrations exist.
- Tool parser is Pythonic LFM2.
- Reasoning default is none unless capability stamp says otherwise.

FalconH1 / Jamba / BaichuanM1 / MiMo:

- CacheList and Mamba/rotating composite paths matter.
- Compile eligibility depends on sub-cache readiness.
- Disk serializer v2 can store `CacheList` sublayers, but live family rows still
  need per-model proof before production.

## Production-Proof Checklist For Any Model Row

For every model/family, a real pass needs:

- Exact local bundle path and model type.
- Prompt and decode token/s.
- Activity Monitor `phys_footprint` when compression/routed residency is part
  of the claim.
- Visible output first/last chars.
- Stop reason and no-loop verdict.
- Reasoning channel verdict: off, on, effort levels, close status.
- Tool-call verdict if supported: schema, no plaintext leak, tool-result turn.
- Multi-turn same-session proof.
- Cache stats: prefix, paged, disk L2, SSM companion, media salt, and
  architecture-specific state as applicable.
- KV policy: none/affine/TurboQuant, maxKVSize, rotating/sliding behavior.
- Scheduler mode: solo, batch, compiled, uncompiled, or hybrid fallback.
- JANG/JANGTQ mode: affine, codebook, streaming diagnostic, MLXPress/JangPress
  residency, sidecar bits.
- Media proof for VL/video/audio rows using real payloads.
- Failure-mode honesty: loops, unclosed reasoning, hidden reasoning-only output,
  EOS leakage, length stops, vector drift, OOM, or cache miss must be recorded.
