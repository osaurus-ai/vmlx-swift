# Nemotron Omni Audio Latency Bench - 2026-05-12

Host: local M5 Max MacBook.

Model: `Nemotron-Omni-Nano-JANGTQ-CRACK`

Bench executable: `swift run OmniAudioLatencyBench`

2026-05-17 consolidation update: this executable is now registered in the
top-level `vmlx-swift` package. The measurements below are the original
standalone Swift LM evidence; current release-built `vmlx-swift` artifacts are
listed in `docs/VMLX_ACTIVE_MODEL_PRODUCTION_SCOPE_2026_05_17.md`.

2026-05-17 current-source correction: the bench now constructs
`GenerateParameters` from `context.configuration.generationDefaults` and prints
the resolved sampling row. The current JANGTQ artifact shows
`temperature=0.600`, `top_p=0.950`, `top_k=0`, `min_p=0.000`, and
`repetition_penalty=1.000`; no greedy temperature override is applied unless a
caller adds one explicitly outside this bench.

## Command

```sh
BENCH_OMNI_AUDIO_PATH=batch \
BENCH_OMNI_AUDIO_PREENCODE=1 \
BENCH_OMNI_AUDIO_DISK_CACHE=1 \
BENCH_AUDIO_REPEATS=2 \
BENCH_AUDIO_FILE=Tests/MLXLMTests/Resources/audio_only.mov \
BENCH_MAX_TOKENS=8 \
BENCH_MODEL=<absolute path to Nemotron-Omni-Nano-JANGTQ-CRACK> \
/usr/bin/time -l .build/debug/OmniAudioLatencyBench
```

## Result

Audio fixture: `Tests/MLXLMTests/Resources/audio_only.mov`, 80,620 samples,
5,038.8 ms at 16 kHz.

Load: 2,289.1 ms, RSS 5,267.2 MiB.

Pre-encode: 1,335.8 ms for 63 audio tokens at hidden size 2,688.

Prompt topology: 94 prompt tokens, 63 media placeholder tokens, media positions
12...74, `cache_block_size=64`, and 11 media placeholder tokens in the suffix
after the 64-token cache boundary. `prompt_minus_one_after_media=true`, but
the default live-call path does not run an extra prompt-minus-one media
re-prefill because that pushed first-turn latency up in testing and disk
restore was slower than a short pre-encoded forward.

| Path | Mode | Turn | First semantic delta | Total | Tokens | Effective tok/s | RSS MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BatchEngine | raw PCM | 1 | 1,514.1 ms | 1,633.6 ms | 8 | 4.9 | 5,330.3 | 5,294.2 |
| BatchEngine | raw PCM | 2 | 1,498.6 ms | 1,619.8 ms | 8 | 4.9 | 5,348.2 | 5,330.7 |
| BatchEngine | pre-encoded Parakeet | 1 | 208.9 ms | 317.6 ms | 8 | 25.2 | 5,358.9 | 5,348.4 |
| BatchEngine | pre-encoded Parakeet | 2 | 201.8 ms | 320.3 ms | 8 | 25.0 | 5,359.0 | 5,358.9 |

`/usr/bin/time -l`: 8.29 s real, max RSS 7,749,156,864 bytes, swaps 0,
peak memory footprint 15,140,248,656 bytes.

## Read

This bench measures first semantic text delta, not output TTS first audio
byte. Raw PCM is still over the awkward-pause threshold on this fixture
because Parakeet audio encoding and multimodal prefill happen after endpoint.
Passing pre-encoded Parakeet embeddings into `UserInput.Audio.preEncoded`
brings first semantic delta to about 200 ms on the Osaurus BatchEngine path.

Turn 2 raw PCM is still roughly 1.5 s. Do not count raw identical-audio repeat
as a solved conversational prefix-cache path; the 64-token cache block splits
the 63-token audio placeholder run, and exact full hybrid hits are intentionally
skipped because re-feeding the last token would double-count recurrent state.
The live-call path should stream/accumulate Parakeet embeddings while the
caller is speaking and submit pre-encoded audio at endpoint.
