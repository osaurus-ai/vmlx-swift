# Nemotron-Labs Audex quant release and Osaurus wiring sheet

Date: 2026-07-20

Status: **PARTIAL overall; six instruct-mode audio-input/text-output bundles pass.**

This sheet is the release contract for the OsaurusAI 4-, 6-, and 8-bit MLX
affine variants of NVIDIA Nemotron-Labs-Audex-2B and
Nemotron-Labs-Audex-30B-A3B. It records the exact source revisions, conversion
layout, live runtime evidence, Osaurus/Dinoki wiring contract, and known failed
rows. A bundle is not represented as supporting output audio, default-thinking
audio responses, or proven persistent cache reuse.

Companion artifacts:

- `docs/AUDEX-FAMILY-QUANT-RELEASE-2026-07-20.json`
- `docs/AUDEX-FAMILY-PR-COMMENT-2026-07-20.md`
- `docs/AUDEX-2B-OSAURUS-INTEGRATION-SPEC-2026-07-20.md`
- `docs/internal/live-gates/20260720T_audex_quant_release/`
- `scripts/convert-audex-affine.py`
- `scripts/verify-audex-bundle.py`
- `Libraries/MLXVLM/Models/Audex.swift`

Pull requests:

- vMLX runtime: https://github.com/osaurus-ai/vmlx-swift/pull/168
- Osaurus capability and pin wiring: https://github.com/osaurus-ai/osaurus/pull/2116

## 1. Exact sources and destination repositories

| Family | Upstream source | Exact revision | Native `model_type` |
| --- | --- | --- | --- |
| 2B dense | `nvidia/Nemotron-Labs-Audex-2B` | `d43e996bab673833ffb56dcfcc5b658f229f7343` | `nemotron_dense_audex` |
| 30B MoE, 3B active | `nvidia/Nemotron-Labs-Audex-30B-A3B` | `00f0afa02e8ec0a9afc88221e456a02591bfca4c` | `nemotron_h_audex` |

Destination model IDs:

| Family | 4-bit | 6-bit | 8-bit |
| --- | --- | --- | --- |
| 2B | `OsaurusAI/Nemotron-Labs-Audex-2B-4bit` | `OsaurusAI/Nemotron-Labs-Audex-2B-6bit` | `OsaurusAI/Nemotron-Labs-Audex-2B-8bit` |
| 30B-A3B | `OsaurusAI/Nemotron-Labs-Audex-30B-A3B-4bit` | `OsaurusAI/Nemotron-Labs-Audex-30B-A3B-6bit` | `OsaurusAI/Nemotron-Labs-Audex-30B-A3B-8bit` |

All six outputs use MLX affine quantization with group size 64. Only the
language decoder is quantized. `audio_encoder.*` and `audio_projector.*`
remain in their source precision.

The model license is the NVIDIA OneWay Noncommercial License. Every uploaded
repository must carry the matching source-family license and third-party
notices. These are not general commercial-use bundles.

## 2. Runtime architecture and quant layout

### 2B dense

- 28-layer Nemotron Dense decoder, hidden width 2,048.
- 16 query heads, 8 KV heads, head dimension 128.
- 32-layer NV-Whisper/Qwen2 audio encoder, width 1,280.
- Audio projector: 1,280 -> 4,096 -> 2,048 with squared ReLU.
- 750 audio embeddings per 30-second 16 kHz clip.
- Standard causal KV cache per language layer.

The converter quantizes 170 two-dimensional decoder/output modules. The 490
audio tensors remain source precision.

### 30B-A3B hybrid MoE

- 52 language blocks, hidden width 2,688.
- Hybrid pattern:
  `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`.
- 23 Mamba blocks, 6 attention blocks, and 23 routed-MoE blocks.
- 128 routed experts, top-6 routing, plus one shared expert.
- 32 query heads, 2 KV heads, head dimension 128.
- Mamba: 64 heads, head dimension 64, state size 128, convolution width 4.
- Same 32-layer NV-Whisper audio encoder as the 2B model.
- Audio projector: 1,280 -> 4,096 -> 2,688 with squared ReLU.

The converter quantizes 118 ordinary language modules and pre-stacks 46 routed
expert groups. Each source group
`backbone.layers.N.mixer.experts.E.{up_proj,down_proj}.weight` becomes one of:

```text
backbone.layers.N.mixer.switch_mlp.fc1.{weight,scales,biases}
backbone.layers.N.mixer.switch_mlp.fc2.{weight,scales,biases}
```

Stacking occurs before quantization, producing the `[128, out, in]` layout
required by `QuantizedSwitchLinear`. Per-expert affine tensors are not shipped.

The source 30B config contains the non-JSON literal `Infinity` in
`time_step_limit`. The converter omits that field and records the change in
`vmlx_conversion.json`; `NemotronHConfiguration` then applies its existing
`[0.001, +infinity]` runtime semantics. No other model behavior is substituted.

## 3. Static bundle verification

`scripts/verify-audex-bundle.py` reads every safetensors header and proves:

- every indexed shard exists;
- each tensor appears exactly once;
- every index entry points to the shard that owns the tensor;
- actual weight bytes equal both index and conversion metadata;
- each quantized module has a complete weight/scales/biases triplet;
- audio encoder and projector tensors have no affine scales/biases;
- every audio tensor has the same dtype, shape, and raw-data SHA-256 as the
  exact pinned source snapshot;
- 30B outputs contain all 46 expected stacked expert groups and no unstacked
  routed-expert tensors;
- no `.incomplete` or `.lock` files are included.

| Bundle | Shards | Weight bytes | Tensors | Quantized modules | Stacked expert groups | Audio tensors | Static result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2B 4-bit | 3 | 2,567,319,984 | 1,057 | 170 | 0 | 490 | PASS |
| 2B 6-bit | 4 | 3,129,880,955 | 1,057 | 170 | 0 | 490 | PASS |
| 2B 8-bit | 4 | 3,692,442,115 | 1,057 | 170 | 0 | 490 | PASS |
| 30B 4-bit | 26 | 19,306,435,800 | 1,219 | 164 | 46 | 490 | PASS |
| 30B 6-bit | 50 | 27,298,484,028 | 1,219 | 164 | 46 | 490 | PASS |
| 30B 8-bit | 50 | 35,290,531,729 | 1,219 | 164 | 46 | 490 | PASS |

Machine-readable reports are under
`docs/internal/live-gates/20260720T_audex_quant_release/`.

## 4. Live runtime matrix

Hardware evidence came from the current local Apple-silicon runtime. The proof
uses the built `Gemma4AudioSmoke` executable but exercises the general
`MLXLMCommon.loadModel` and production `MLXLMCommon.generate` stream. Each PASS
row used:

- the exact local quant bundle;
- `sample_speech.wav`, decoded and resampled through `AudexProcessor`;
- 96-token maximum as a safety budget, with `.stop` required (a length stop is
  a failure);
- bundle generation defaults: temperature 0.6, top-p 1.0, top-k 0, min-p 0,
  and no repetition penalty;
- explicit request-level `enable_thinking=false` to select NVIDIA's documented
  instruct template mode;
- an audio transcription turn followed by two growing-history questions;
- non-empty visible output, zero reasoning-channel characters, no protocol
  marker leakage, measured token/s on every turn, and sampled
  `TASK_VM_INFO.phys_footprint`.

The audio says that stew includes turnips, carrots, bruised potatoes, and
mutton. Turn 2 must name at least two vegetables. Turn 3 must confirm that
mutton was mentioned.

| Bundle | Load | Audio turn | Turn 2 | Turn 3 | Current footprint | Peak sampled footprint | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2B 4-bit | 4.8 s | 117.1 tok/s | 119.6 tok/s | 114.7 tok/s | 3.868 GiB | 6.427 GiB | PASS instruct |
| 2B 6-bit | 4.6 s | 117.6 tok/s | 114.9 tok/s | 113.8 tok/s | 4.365 GiB | 6.413 GiB | PASS instruct |
| 2B 8-bit | 4.6 s | 121.8 tok/s | 114.8 tok/s | 122.1 tok/s | 4.839 GiB | 6.897 GiB | PASS instruct |
| 30B 4-bit | 4.7 s | 50.8 tok/s | 50.6 tok/s | 51.3 tok/s | 20.444 GiB | 24.127 GiB | PASS instruct |
| 30B 6-bit | 4.9 s | 50.0 tok/s | 51.1 tok/s | 49.8 tok/s | 27.873 GiB | 31.452 GiB | PASS instruct |
| 30B 8-bit | 5.2 s | 49.5 tok/s | 50.2 tok/s | 50.2 tok/s | 35.367 GiB | 38.944 GiB | PASS instruct |

These footprint numbers are direct samples, not low-RAM claims. Prompt time,
prefill throughput, energy, and disk-read pressure were not recorded in this
matrix.

### Required negative evidence

Omitting `enable_thinking` selects the source template's default thinking
mode. On representative 2B 4-bit and 30B 4-bit rows, the models emitted a
coherent transcription but did not close the pre-opened reasoning envelope.
The production parser therefore emitted 195-196 reasoning characters, zero
visible characters, and a normal stop. Those rows are FAIL for user-visible
chat.

Do not fix this by forcing a closing tag, exposing unclosed reasoning as final
text, changing sampling defaults, or coercing the prompt. Until thinking-mode
audio is corrected and live-proven, Osaurus/Dinoki must send the user's
explicit instruct choice as `enable_thinking=false` for these audio requests
and must not advertise default-thinking audio as passing.

## 5. Osaurus and Dinoki wiring contract

### Discovery and capability

Both folder/repository names and config model types must resolve to audio-only:

```text
name contains nemotron-labs-audex or nemotron_labs_audex
config.model_type in {nemotron_dense_audex, nemotron_h_audex}
```

The resulting UI/runtime capability is:

```swift
Capabilities(supportsImage: false, supportsVideo: false, supportsAudio: true)
```

Audex loads through the VLM factory because audio embeddings are spliced into
the language decoder. That factory identity must not trigger a generic image
attachment fallback.

### Request construction

For the proven instruct path:

```swift
let input = UserInput(
    chat: [
        .user(
            "Transcribe the speech accurately.",
            audios: [.url(audioURL)])
    ],
    additionalContext: ["enable_thinking": false]
)
```

The production request must retain the bundle's generation defaults unless the
user explicitly chooses a task override. NVIDIA recommends greedy decoding for
ASR/translation and temperature 0.7/top-p 0.9 for audio understanding; those
are task recipes, not family-wide hidden defaults.

The OpenAI-compatible route should map `input_audio` or `audio_url` content to
`Chat.Message.audios`, preserve transcript/history messages, and call the
normal VLM processor. Do not inject literal `<sound>` or
`<so_embedding>` text from the host; `AudexProcessor` owns placeholder
expansion.

### Input/output boundary

| Surface | Status |
| --- | --- |
| Text plus audio -> text | PASS in instruct mode |
| Growing audio conversation history -> text | PASS in instruct mode |
| Default-thinking audio -> visible text | FAIL, unclosed reasoning |
| Image input | Unsupported; reject before inference |
| Video input | Unsupported; reject before inference |
| Batch size > 1 | Unsupported by native wrapper; schedule batch size 1 |
| Text-to-speech output | Not implemented in this Swift runtime |
| Text-to-audio output | Not implemented in this Swift runtime |
| Speech-to-speech audio output | Not implemented in this Swift runtime |

Do not advertise output-audio features merely because the upstream repository
contains audio-generation checkpoints and codecs. These quants contain the
full audio-input checkpoint only.

## 6. Cache contract and open proof

The 2B model owns 28 standard causal KV caches. The 30B model owns 29
cache-bearing states: 23 `MambaCache` entries and 6 attention KV caches; its 23
MoE blocks do not own KV state.

The live quant matrix above uses growing chat history but does not claim a
prefix, paged, L2-disk, TurboQuant-KV, or audio-embedding cache hit. Before a
release can claim cache persistence, prove and record:

1. exact prefix-match counts on normalized PCM/media-salted prompts;
2. Mamba companion-state restoration at the matched boundary for 30B;
3. attention KV restore for the six attention layers;
4. paged and L2-disk counters, including a fresh-process restore;
5. coherent visible output after the hit;
6. whether NV-Whisper/projector embeddings are recomputed or reused.

`turbo_quant_kv_layer_count=0` must be reported as zero; it must not be
described as a TurboQuant-KV topology.

## 7. Reproduction commands

Convert:

```bash
uv run --project ../jang/jang-tools python scripts/convert-audex-affine.py \
  ~/models/nvidia/Nemotron-Labs-Audex-2B/checkpoint_folder_full \
  ~/models/OsaurusAI/Nemotron-Labs-Audex-2B-6bit \
  --bits 6 --group-size 64 \
  --source-model nvidia/Nemotron-Labs-Audex-2B \
  --source-revision d43e996bab673833ffb56dcfcc5b658f229f7343
```

Validate headers/index:

```bash
python3 scripts/verify-audex-bundle.py \
  ~/models/OsaurusAI/Nemotron-Labs-Audex-2B-6bit \
  --expect-bits 6 \
  --source ~/models/nvidia/Nemotron-Labs-Audex-2B/checkpoint_folder_full
```

Run the proven production-stream row:

```bash
GEMMA4_SMOKE_MODEL=~/models/OsaurusAI/Nemotron-Labs-Audex-2B-6bit \
GEMMA4_SMOKE_AUDIO=~/models/nvidia/Nemotron-Labs-Audex-Space/examples/sample_speech.wav \
GEMMA4_SMOKE_PROMPT='Transcribe the speech accurately.' \
GEMMA4_SMOKE_MAX_TOKENS=96 \
GEMMA4_SMOKE_MULTI_TURN=1 \
GEMMA4_SMOKE_ENABLE_THINKING=false \
.build/arm64-apple-macosx/debug/Gemma4AudioSmoke
```

Download after Hub publication:

```bash
hf download OsaurusAI/Nemotron-Labs-Audex-2B-4bit \
  --local-dir ~/models/OsaurusAI/Nemotron-Labs-Audex-2B-4bit
```

Replace the repository ID with any of the six IDs in section 1.

## 8. Release gate

Before marking an uploaded repository usable:

1. Confirm the Hub repository is public and the model card renders.
2. Confirm `config.json`, tokenizer, template, generation config,
   `vmlx_conversion.json`, license, notices, index, and every indexed shard are
   present.
3. Re-download or dry-run the Hub revision and verify no file is missing.
4. Run `verify-audex-bundle.py` on the downloaded revision with the exact
   pinned source snapshot supplied through `--source`.
5. Run the exact production-stream audio conversation proof.
6. Record source and Hub commit SHAs in the release JSON.

Until the Hub round-trip and exact Osaurus app pin are complete, this document
remains **PARTIAL**, even though the six local instruct-mode rows pass.
