# Nemotron-Labs Audex-2B vMLX checkpoint — 2026-07-20

Integration handoff:

- `docs/AUDEX-2B-OSAURUS-INTEGRATION-SPEC-2026-07-20.md`
- `docs/AUDEX-2B-OSAURUS-WIRING-MANIFEST.json`
- `docs/AUDEX-2B-PR-COMMENT-2026-07-20.md`
- `docs/internal/live-gates/20260720T_audex_2b_vmlx/README.md`

## Scope and artifacts

- Source model: `~/models/nvidia/Nemotron-Labs-Audex-2B`
  at Hugging Face revision `d43e996bab673833ffb56dcfcc5b658f229f7343`.
- Runtime checkpoint: `checkpoint_folder_full` inside the source model.
- Official Space snapshot and audio fixtures:
  `~/models/nvidia/Nemotron-Labs-Audex-Space`, revision
  `c66674198bafcc086730538b1d0c86b759b133ee`.
- vMLX 4-bit affine bundle:
  `~/models/nvidia/Nemotron-Labs-Audex-2B-4bit-vMLX`.
  Dense text weights are 4-bit/group-64; NV-Whisper and the audio projector
  remain at source precision. The two weight shards total 2.391 GiB.

## Correctness checkpoint

| Gate | Status | Evidence |
| --- | --- | --- |
| Complete source/Space download | PASS | 106 model payload files, 12,281,588,752 bytes; Hugging Face dry-run reported no missing download. Space snapshot is 531 MiB. |
| Native architecture load | PASS | `model_type=nemotron_dense_audex` dispatches to `Audex`; source checkpoint loaded in 4.7 s and quantized bundle in 4.3–4.5 s. |
| Bundle-driven tokenizer/template | PASS | Official tokenizer and chat template are used. Reserved indexed codec tokens stay addressable by ID but are excluded from the ICU added-token alternation; `prepare` fell from about 40 s to 44–48 ms. |
| Exact audio frontend | PASS | 16 kHz, 30 s right padding, periodic Hann STFT, Slaney mel filters, Whisper log normalization, NV-Whisper encoder, pooling, and projector are native Swift/MLX code. Deterministic feature fixtures come from the official processor. |
| Source audio-in/text-out | PASS | Official `sample_speech.wav` produced an accurate stew/turnips transcript, 48 tokens at 45.9 tok/s. |
| Quantized audio-in/text-out | PASS | The same file produced an accurate transcript, 46 tokens at 43.6 tok/s. Official `mlk_speech.wav` produced the expected “I have a dream…” transcript, 33 tokens at 43.2 tok/s. No protocol markers leaked. |
| Physical footprint | PASS for measured audio row | `/usr/bin/footprint` observed `phys_footprint_peak=3,073,313,624` bytes during the quantized live audio run. This was measured physical footprint, not an MLX memory-limit throttle. Raw output is preserved under `docs/internal/live-gates/20260720T_audex_2b_vmlx/`. |
| Multi-turn text/cache path | PASS | Quantized bundle, three turns, compile off and on: remembered “blue”, then answered “Cool.” TTFT was 66–116 ms. Standard per-layer KV caches were used. |
| Capability status | PASS in vMLX source | `ModelRuntimeCapabilitySnapshot` reports audio supported for `nemotron_dense_audex`; vision/video remain unclaimed without explicit bundle capability metadata. |
| Focused tests | PASS | Xcode-hosted Swift Testing: Audex feature parity and placeholder expansion 2/2; capability snapshot 1/1; tokenizer placeholder guard 1/1. The test executable required the generated `mlx.metallib` to be colocated in its build bundle. |
| Osaurus composer/source wiring | PARTIAL | Generic Osaurus audio attachments already map to vMLX audio input. A separate Osaurus worktree adds Audex as audio-only by name and `model_type`; the production target compiled, a direct source smoke test passed all three ID forms plus bundle `config.json`, and all three focused Osaurus tests passed. The current package pin and packaged-app UI proof are still outstanding. |
| Osaurus focused tests | PASS | Xcode-hosted Swift Testing ran the name-based, bundle-`model_type`, and `isVLM=true` image-fallback capability cases: 3 tests in 3 suites passed. The full Xcode developer directory was required because the selected CommandLineTools SDK does not expose the `Testing` framework. |
| Audio generation (TTS/text-to-audio/speech-to-speech) | BLOCKED | The checkpoint's separate causal speech decoder and XCodec output path are downloaded but not implemented in vMLX. No output-audio claim is made. |

## Implemented runtime surface

- Native dense Audex decoder with GQA, RoPE, RMSNorm, and ReLU-squared MLP.
- Native NV-Whisper/Qwen2 audio encoder and audio-to-text projector.
- Raw PCM and pre-encoded audio input, long-audio 30-second splitting, typed
  rejection of image/video, media cache salt, and chunked embedded prefill.
- Standard vMLX per-layer KV caching and model-load/prefill progress callbacks.
- Source and affine quantized loading through the same VLM factory and
  generation path; no prompt, sampler, reasoning-tag, or decode-loop coercion.
- Converter: `scripts/convert-audex-affine.py`. It refuses to overwrite a
  non-empty output directory.

## Remaining proof before an Osaurus-ready claim

1. Land this vMLX branch and update the exact Osaurus package pin.
2. Land the Audex audio-only capability matcher in Osaurus.
3. Build and launch the development Osaurus app, select the local 4-bit
   bundle, attach a real WAV file in Chat/Dinoki, and capture visible output,
   tok/s, and physical-footprint evidence through the app-owned process.
4. Exercise the OpenAI-compatible audio-bearing chat route and a second turn.
5. Implement and independently prove the causal speech decoder plus XCodec
   before advertising TTS, text-to-audio, or speech-to-speech.

This checkpoint is therefore runtime-capable for the tested audio-to-text and
text-chat lanes inside vMLX, but it is not yet a full all-modality Audex or
packaged Osaurus completion.
