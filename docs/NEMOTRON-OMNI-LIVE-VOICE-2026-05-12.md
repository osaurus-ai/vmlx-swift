# Nemotron Omni Live Voice Status - 2026-05-12

## Scope

This note tracks the current live voice path for agentic call-style input:
Nemotron Omni audio input, Parakeet/RADIO encoder handling, prefix-cache
isolation, Osaurus handoff requirements, and measured local behavior.

2026-05-17 consolidation update: the live voice bench executables described
here now live in this package as `OmniAudioLatencyBench` and
`OmniAudioChunkStabilityBench`. The original 2026-05-12 measurements below are
kept as historical evidence; current `vmlx-swift` release-built evidence is
tracked in `docs/VMLX_ACTIVE_MODEL_PRODUCTION_SCOPE_2026_05_17.md`.

2026-05-17 08:56 PDT current-checkout recheck: release builds and live audio probes were rerun
from this `vmlx-swift` checkout under
`docs/local/live-model-matrix/20260517T155603Z_omni_live_voice_current_verify/`.
`OmniAudioLatencyBench`, `OmniAudioChunkStabilityBench`, and `RunBench` all
rebuilt in release mode. The Xcode-backed focused test command passed 8/8; a
plain CLI `swift test` invocation can still fail to locate the Swift `Testing`
module on this machine, so use the documented `DEVELOPER_DIR=... xcrun swift
test ... -Xswiftc -F .../Frameworks` form for current proof.

Fresh live evidence:

- `Nemotron-Omni-Nano-JANGTQ4-CRACK` full Omni `RunBench` at 48 tokens passed
  18/18 rows with bundle generation defaults, including text, image, video,
  audio, mixed image+audio, media-salt isolation, hybrid SSM warm-pass, and
  BatchEngine text/image/audio rows. Load was 1.79 s; direct decode rows were
  88.4-110.3 tok/s and BatchEngine rows were 37.6-70.8 tok/s.
- The JANGTQ4 live audio bench used `temperature=0.600`, `top_p=0.950`,
  `top_k=0`, `min_p=0.000`, `repetition_penalty=1.000` from
  `generation_config.json`; it pre-encoded Parakeet audio to `63 x 2688`
  embeddings in 50.1 ms. Raw PCM and pre-encoded audio both streamed through
  BatchEngine and TokenIterator. First deltas were 203.5-219.3 ms for raw
  BatchEngine, 176.0-188.7 ms for pre-encoded BatchEngine, 184.6-188.5 ms for
  raw TokenIterator, and 157.1-157.7 ms for pre-encoded TokenIterator.
- The JANGTQ and MXFP4 Omni bundles also loaded and streamed the same fixture
  through raw/pre-encoded BatchEngine and TokenIterator paths at 32 tokens.
  JANGTQ pre-encode was 43.9 ms; MXFP4 pre-encode was 48.1 ms.
- Disk/SSM cache artifacts were written for the audio bench paths:
  `cache_index.db`, safetensors entries, and `ssm_companion` directories are
  listed in `cache_artifacts_listing.txt`.
- `OmniAudioChunkStabilityBench` confirms independently encoded Parakeet chunks
  are still not concat-safe: 10/10 prefix comparisons were unstable at the
  default tolerance. The production live voice contract remains retained PCM
  plus refreshed full-snapshot pre-encode, or raw PCM at endpoint. Do not stitch
  independently encoded chunk embeddings into the model context.
- Coherency caveat: the audio answers are grounded in the fixture, but several
  48-token rows repeat a short sentence or continue instead of cleanly stopping.
  This is a visible runtime/termination boundary; do not hide it with sampler
  clamps or forced stop guards.

2026-05-17 09:31 PDT recheck under
`docs/local/live-model-matrix/20260517T163112Z_omni_live_voice_reverify_current/`:
the focused 8-test Omni/Parakeet suite passed, the latency and chunk-stability
benches rebuilt in release mode, and JANGTQ4 passed the integrated 18/18
`BENCH_OMNI=1 BENCH_OMNI_BATCH=1` matrix at 48 tokens. The cleanest live voice
path is currently cache OFF or fresh prompt-boundary reuse: repeated cache-off
raw/pre-encoded BatchEngine and TokenIterator rows were grounded with no marker
leak. Repeated disk-cache ON audio remains partial because one sampled
pre-encoded TokenIterator cache-reuse row emitted sound marker text and a few
sampled rows were weak. Keep this as a cache-hit output-quality/root-cause item,
not a reason to add hidden sampler clamps, forced stop tokens, or post-hoc text
cleanup.

2026-05-17 10:06 PDT source-wrapper fix under
`docs/local/live-model-matrix/20260517T170614Z_omni_live_voice_fresh_recheck/`:
the Swift processor had been wrapping audio slots as literal `<sound>` and
`</sound>` text. The bundled processor uses `<so_start>` and `<so_end>` around
the repeated `<so_embedding>` slots. The new focused regression failed before
the fix with literal sound-marker token `95690` and missing wrapper token IDs
`28`/`29`, then passed after the processor emitted the source-compatible wrapper.
The full focused suite now passes 9/9. Cache-off live audio after the fix uses
bundle defaults, pre-encodes the fixture to `63 x 2688` in 46.7 ms, streams all
raw/pre-encoded BatchEngine and TokenIterator paths at 65.4-76.1 tok/s, and no
longer emits literal sound-marker text. A 12-row cache-off repeat stayed marker
clean. Short BatchEngine/pre-encoded rows can still be weak, so this is a real
token-wrapper fix but not a claim that every short stochastic audio row is
quality-complete.

2026-05-17 10:43 PDT fresh post-fix integration proof under
`docs/local/live-model-matrix/20260517T174343Z_omni_parakeet_fresh_verify/`:
the current checkout passes the Xcode-backed focused Omni/Parakeet suite 9/9,
release-builds `OmniAudioLatencyBench` and `RunBench`, and reloads the local
`Nemotron-Omni-Nano-JANGTQ4-CRACK` bundle. The live audio bench uses bundle
defaults (`temperature=0.600`, `top_p=0.950`, `top_k=0`, `min_p=0.000`,
`repetition_penalty=1.000`), pre-encodes Parakeet audio to `63 x 2688` in
45.8 ms, and streams all four paths:

- BatchEngine raw PCM: first delta 223.6 ms, 64.5 tok/s, grounded chime/beep
  description.
- BatchEngine pre-encoded Parakeet: first delta 169.5 ms, 72.4 tok/s.
- TokenIterator raw PCM: first delta 183.9 ms, 68.9 tok/s.
- TokenIterator pre-encoded Parakeet: first delta 151.6 ms, 74.8 tok/s.

The same artifact records prompt topology of 93 prompt tokens, 63 audio
placeholder tokens, media token ids `[18, 27]`, and 9 media tokens after the
64-token cache boundary. The integrated `BENCH_OMNI=1 BENCH_OMNI_BATCH=1`
`RunBench` row passes 18/18 at 48 tokens, including BatchEngine audio B=1 and
hybrid SSM warm-pass parity. This confirms the live voice input path is real
for cache-off/fresh prompt-boundary use. Cache-on repeated live audio remains a
separate output-quality/root-cause item; keep it honest, not patched over with
hidden sampling or text cleanup.

## Implemented

- `UserInput.Audio.preEncoded(samples:sampleRate:embedding:)` exists for live
  voice handoff when another component has already produced Parakeet/sound
  projection embeddings.
- `NemotronHOmniProcessor` preserves pre-encoded audio embeddings while still
  carrying the waveform for media salt and fallback behavior. Audio placeholders
  are wrapped with the bundled processor's `<so_start>`/`<so_end>` tokens, not
  literal `<sound>` text.
- `NemotronHOmni.prepare(_:)` uses `audio.preEncodedEmbedding` when present and
  otherwise runs the mel + Parakeet + sound projection encoder path.
- The Python vMLX Omni dispatcher hashes user media bytes/paths into its
  session signature. Same text plus different prior audio/image/video now
  resets the Omni session instead of reusing stale media-conditioned state.
- Osaurus now retains raw live voice PCM separately from the STT worker drain
  and attaches a WAV to voice sends when the selected model advertises audio.

## Bench Evidence

Command:

```sh
BENCH_OMNI=1 \
BENCH_MAX_TOKENS=8 \
BENCH_MODEL=<local Nemotron-Omni-Nano-JANGTQ-CRACK> \
/usr/bin/time -l swift run RunBench
```

Artifact:

```text
build/evidence-20260512/nemotron_omni_live_voice_jangtq2_clean_20260512_155446.log
```

Inputs:

- Model: local `Nemotron-Omni-Nano-JANGTQ-CRACK` bundle, `weight_format=mxtq`,
  `mxtq_bits=2`, `config_omni.json` present.
- Audio fixture: `Tests/MLXLMTests/Resources/audio_only.mov`.

Result summary:

| Row | Result | Time | Throughput |
|---|---:|---:|---:|
| text-only single-turn | PASS | 0.23s | 61.5 tok/s |
| text-only multi-turn x3 | PASS | 0.55s | 62.1 tok/s |
| image single-turn | PASS | 1.57s | 43.5 tok/s |
| image multi-turn x2 | PASS | 3.01s | 44.5 tok/s |
| video encoder smoke | PASS | 0.33s | n/a |
| audio encoder smoke | PASS | 0.29s | n/a |
| video LMInput end-to-end | PASS | 15.22s | 13.1 tok/s |
| audio LMInput end-to-end | PASS | 1.56s | 52.3 tok/s |
| reasoning OFF | PASS | 0.19s | 60.2 tok/s |
| reasoning ON/OFF/ON toggle | PASS | 0.52s | 57.5 tok/s |
| mixed image + audio | PASS | 2.88s | 41.0 tok/s |
| media-salt isolation audio A vs B | PASS | 2.18s | 56.0 tok/s |
| hybrid SSM warm-pass parity | PASS | 0.37s | 61.1 tok/s |

Overall: `13 passed, 0 failed`, `bench_exit=0`, load time `2.37s`.

Memory:

- System memory free before: 93%.
- System memory free after: 93%.
- Maximum resident set size: `7,740,358,656` bytes.
- Peak memory footprint reported by `/usr/bin/time -l`: `79,949,715,888`
  bytes.
- Swaps during command: `0`.

## Latency Interpretation

This bench proves real audio input encoding/splice/generation and multi-turn
cache behavior. It is not yet a TTFAB benchmark because it does not synthesize
or stream output audio.

Current useful input-side measurements:

- Parakeet encoder smoke for the fixture: `0.29s`.
- Full audio `LMInput` end-to-end row with 8-token decode: `1.56s`.
- Mixed image + audio row with 8-token decode: `2.88s`.

For live calls, the `preEncoded` path is important because it can remove the
mel + Parakeet encoder cost from the model turn if a streaming voice component
has already produced embeddings. The remaining unresolved latency question is
not "can Omni consume audio" but "can we produce stable streaming audio
embeddings and TTS first audio fast enough for natural turn-taking."

## Remaining Hookups

- Add a true TTFAB bench: audio input clip -> model first visible token ->
  TTS first audio byte. Record VAD endpoint delay, model first token, TTS
  first audio byte, and total wall time separately.
- Add an Osaurus app-level live voice trace that logs:
  `voice_snapshot_ms`, `wav_bytes`, `input_audio_materialize_ms`,
  `prompt_prepare_ms`, `first_token_ms`, and response stream first chunk.
- Add a direct `preEncoded` route from an embedding-producing voice component
  into `UserInput.Audio.preEncoded`. The current Osaurus patch sends a WAV
  attachment, which is correct but still pays model-side audio encoding.
- Keep prefix/cache keys media-aware. Any future embedding-only path must hash
  the original waveform or a stable embedding digest in addition to text.
- Add TTS TTFA/TTFAB measurement. Nemotron Omni does not include a neural TTS
  decoder in this runtime; call output speech must be handled by a separate TTS
  backend.
- Fix or route around local Swift test-runner blockers before claiming unit
  tests are green:
  - `vmlx-swift-lm`: `swift test` currently fails on `no such module 'XCTest'`.
  - `osaurus-staging`: `swift test` currently fails on `no such module 'Testing'`.

## Library Boundary

- `vmlx-swift-lm`: code changes are required and implemented for pre-encoded
  audio and Omni prepare.
- `vmlx_engine` Python: code changes are required and implemented for
  media-aware Omni session signatures and `audio/mp4 -> .m4a`.
- `osaurus-staging`: code changes are required and implemented for retaining
  live voice PCM and attaching WAV audio to audio-capable voice sends.
- `swift-jinja`, `swift-transformers`, and `mlx-swift`: no direct code changes
  were required in this pass. Existing package pins should still be checked
  before PR merge.
