# Nemotron Omni Audio Chunk Stability Bench - 2026-05-13

Host: local M5 Max MacBook.

Model: `Nemotron-Omni-Nano-JANGTQ-CRACK`

Bench executable: `swift run OmniAudioChunkStabilityBench`

2026-05-17 consolidation update: this executable is now registered in the
top-level `vmlx-swift` package. The measurements below are the original
standalone Swift LM evidence; current release-built `vmlx-swift` artifacts are
listed in `docs/VMLX_ACTIVE_MODEL_PRODUCTION_SCOPE_2026_05_17.md`.

2026-05-17 current-source rerun:
`docs/local/live-model-matrix/20260517T_omni_live_voice_current/omni_audio_chunk_stability.log`
keeps the same contract. Full retained-audio Parakeet encode produced 63
tokens in about 48 ms, every prefix/full comparison still had
`stable_tokens_default=0`, and independently encoded chunks remain unsafe to
concatenate into the model context.

## Command

```sh
BENCH_MODEL=<absolute path to Nemotron-Omni-Nano-JANGTQ-CRACK> \
BENCH_AUDIO_FILE=<16 kHz WAV fixture> \
BENCH_CHUNK_SECONDS=1,2,3,4,5 \
BENCH_STABILITY_TOLERANCE=0.01 \
/usr/bin/time -l swift run OmniAudioChunkStabilityBench
```

## Result

Audio fixture: 80,620 samples, 5,038.8 ms at 16 kHz.

Load: 2,272.3 ms, RSS 5,265.5 MiB.

Full Parakeet encode: 1,311.3 ms for 63 audio tokens at hidden size 2,688.

Prefix Parakeet encodes:

| Prefix | Samples | Audio tokens | Encode ms |
|---:|---:|---:|---:|
| 1 s | 16,000 | 13 | 276.1 |
| 2 s | 32,000 | 26 | 529.9 |
| 3 s | 48,000 | 38 | 774.5 |
| 4 s | 64,000 | 51 | 1,014.1 |
| 5 s | 80,000 | 63 | 1,259.2 |

Prefix comparison summary at default tolerance 0.01:

| Comparisons | Unstable comparisons | Chunk concat safe |
|---:|---:|---|
| 10 | 10 | no |

Every prefix comparison had `stable_tokens_default=0`. That remained true
against both the final full clip and the next-longer prefix. It also remained
unsafe at tolerances 0.1, 0.01, 0.001, and 0.0001.

`/usr/bin/time -l`: 13.24 s real, max RSS 7,747,502,080 bytes, swaps 0,
peak memory footprint 15,139,773,592 bytes.

## Read

The current Parakeet path is not prefix-stable. Extending the audio changes
earlier audio embeddings, including the first comparable token, so concatenating
independently encoded Parakeet chunks is not a safe live-streaming strategy.

The safe current shape for agentic call input is to stream microphone PCM into a
resident buffer, refresh a full retained-audio Parakeet pre-encode while the
caller is speaking, and submit the latest exact pre-encoded audio snapshot at
endpoint. That removes endpoint-time audio encoding when the warm snapshot is
fresh, while avoiding invalid chunk concatenation.

## Release Gate

For Osaurus live voice work, Parakeet and RADIO code is not considered shipped
until the implementation has been checked in all of these places:

- Source repo contains the Parakeet/RADIO functions and benchmark target.
- Source repo commit is pushed to the live library remote.
- Osaurus pins that exact pushed source commit.
- Osaurus checkout contains the same Parakeet/RADIO functions after package
  resolution.
- Repo-visible docs record the measured Parakeet/RADIO behavior and the safe
  live-call streaming contract.
