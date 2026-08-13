# Audex-2B PR comment package

Copy the section below into the vMLX PR. Replace only the explicit
`REPLACE_*` fields after the branch has been committed and pushed. Do not turn
pending rows into passes without the named evidence.

---

## Nemotron-Labs-Audex-2B native runtime and Osaurus handoff

### What this PR adds

This PR adds native vMLX support for NVIDIA
`nvidia/Nemotron-Labs-Audex-2B`:

- native `nemotron_dense_audex` model loading;
- dense 28-layer Nemotron text decoder with GQA, RoPE, RMSNorm, and
  ReLU-squared MLP;
- native NV-Whisper/Qwen2 audio encoder and ReLU-squared audio projector;
- exact 16 kHz Whisper-compatible 128-bin feature extraction;
- text plus audio input through normal `UserInput`/`LMInput` processing;
- 30-second clip splitting, right padding, and 750 audio embeddings per clip;
- bundle ChatML template, thinking/instruct context, and EOS defaults;
- standard per-layer KV caching with audio-byte/scope/policy cache salting;
- source and MLX affine 4-bit bundle loading through the same factory;
- tokenizer scaling for 74,733 indexed codec/reserved tokens without removing
  their token-ID mappings;
- positive audio capability reporting for `nemotron_dense_audex`;
- focused architecture/frontend/placeholder/capability/tokenizer tests;
- an Osaurus capability patch that exposes Audex as audio-only.

### Exact support boundary

| Lane | Status |
| --- | --- |
| Text input -> text output | PASS |
| Audio input -> text output | PASS |
| 4-bit affine bundle | PASS |
| Multi-turn text/KV behavior | PASS |
| Osaurus audio-only source wiring | PARTIAL: tests pass; final pin/app proof pending |
| Image input | UNSUPPORTED |
| Video input | UNSUPPORTED |
| TTS/text-to-audio/speech-to-speech output | BLOCKED: decoder/XCodec path not implemented |

Audex loads through the VLM factory because it splices audio embeddings. It is
not an image model. Osaurus must resolve it to `audioOnly` and must not widen
that capability through a generic `isVLM` image fallback.

### Pinned source material

- Model: `nvidia/Nemotron-Labs-Audex-2B`
- Model revision: `d43e996bab673833ffb56dcfcc5b658f229f7343`
- Space revision: `c66674198bafcc086730538b1d0c86b759b133ee`
- Source checkpoint: `checkpoint_folder_full`
- Local source snapshot:
  `~/models/nvidia/Nemotron-Labs-Audex-2B`
- Local quant bundle:
  `~/models/nvidia/Nemotron-Labs-Audex-2B-4bit-vMLX`
- Quantized weights: 2,567,320,669 bytes
- Quantization: MLX affine 4-bit, group size 64
- Audio encoder/projector: source precision

The model is governed by the NVIDIA OneWay Noncommercial License and should be
treated as research/development-only in Osaurus catalog and distribution work.

### Architecture and processor facts

| Component | Contract |
| --- | --- |
| Text decoder | width 2048, FFN 9216, 28 layers, Q16/KV8, head dim 128 |
| Vocabulary | 205,312 |
| Context config | 131,072 positions |
| Audio encoder | width 1280, 32 layers, 20 heads, FFN 5120 |
| Projector | 1280 -> 4096 -> 2048, ReLU-squared |
| Audio input | mono Float32 PCM normalized to 16 kHz |
| Clip | 480,000 samples / 30 seconds |
| Feature tensor | `[1, 128, 3000]` per clip |
| Audio embeddings | 750 per clip |
| Wrapper token IDs | embedding 29, start 30, end 31 |
| EOS IDs | 2 and 11 |
| Batch size | 1 |

### Live proof

| Row | Result | Evidence |
| --- | --- | --- |
| Source + `sample_speech.wav` | PASS | loaded 4.7 s; prepare about 46 ms; 48 tokens at 45.9 tok/s; accurate stew/turnips/carrots transcript |
| 4-bit + `sample_speech.wav` | PASS | loaded 4.3 s; prepare 44 ms; 46 tokens at 43.6 tok/s; accurate transcript |
| 4-bit + `mlk_speech.wav` | PASS | loaded 4.5 s; prepare 48 ms; 33 tokens at 43.2 tok/s; correct language and speech transcript |
| Physical footprint | PASS for measured audio row | peak 3,073,313,624 bytes from `/usr/bin/footprint`; raw artifact in `docs/internal/live-gates/20260720T_audex_2b_vmlx/` |
| Three-turn text behavior | PASS | remembered `blue`; visible coherent replies; compile disabled/enabled both passed |
| vMLX focused tests | PASS | audio features and placeholder count 2/2; capability 1/1; tokenizer guard 1/1 |
| Osaurus Audex capability | PASS | three focused tests in three suites: name, bundle `model_type`, and `isVLM=true` image-fallback guard |
| Dinoki/Chat UI | PENDING | requires development app, real WAV attachment, visible answer, tok/s, footprint |
| OpenAI `input_audio` | PENDING | generic path is source-traced; Audex app/server row still required |
| Prefix/paged/L2 cache counters | PENDING | do not infer from source |

No prompt coercion, forced thinking tags, hidden sampler/repetition overrides,
audio-token masking, or synthetic output cap was used as a correctness fix.

### Tokenizer performance fix

The tokenizer includes:

- 970 `<SPECIAL_N>` tokens;
- 65,536 `<speechcodec_N>` tokens;
- 8,192 `<audiocodec_N>` tokens.

These indexed placeholders remain addressable by ID but are excluded from the
ICU added-token alternation. Named special tokens remain in normal isolation.
Measured `processor.prepare` fell from about 39-40 seconds to 44-48 ms.

This is not deletion or masking of codec tokens. Keeping their mappings is
required for bundle fidelity and future output-audio work.

### Cache statement

Audex uses 28 standard dense KV caches. The cache key includes normalized audio
sample rate, waveform shape/dtype/bytes, semantic scope such as thinking mode,
and KV policy. Partial hits roll back to full prefill when token 29 remains in
the suffix, so the audio embedding span cannot be split unsafely.

Proven now:

- normal KV creation;
- coherent three-turn text behavior;
- compiled/non-compiled decode parity;
- media-salt and placeholder-boundary source paths.

Not yet proven:

- Audex audio prefix-cache hit counters;
- paged-cache counters;
- L2 disk restore counters;
- long-audio cache behavior;
- reusable Audex-specific audio embeddings.

Do not describe any of those as active until the Osaurus runtime artifact
records the effective topology and counters.

### Osaurus changes required

1. Merge/push this vMLX PR and capture its exact commit:
   `REPLACE_WITH_PUSHED_AUDEX_VMLX_COMMIT`.
2. Update all Osaurus pin surfaces to that exact SHA:
   - `Packages/OsaurusCore/Package.swift`
   - `Packages/OsaurusCore/Package.resolved`
   - `osaurus.xcworkspace/xcshareddata/swiftpm/Package.resolved`
   - `App/osaurus.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved`
3. Land the prepared Osaurus changes in:
   - `ModelMediaCapabilities.swift`
   - `ChatView.swift`
   - `ModelMediaCapabilitiesMCDCTests.swift`
   - `MultiTurnFamilyMatrixTests.swift`
   - `ModelPickerItemChatCapabilityTests.swift`
4. Keep the model capability exactly `audioOnly`.
5. Reuse existing `input_audio` -> `UserInput.Audio` mapping and normal
   `loadModelContainer`/BatchEngine execution.
6. Build an isolated development Osaurus app and perform the proof matrix
   below.

The current Osaurus pin is
`f2b184841e98d969e46dec83109f27cd7bb57357`; it does not contain Audex.

### Osaurus validation matrix

- [ ] Clean resolve confirms all four pin surfaces use the same Audex vMLX SHA.
- [ ] Local model discovery reads `model_type=nemotron_dense_audex`.
- [ ] Picker and composer enable audio.
- [ ] Picker and composer reject image and video.
- [ ] Switching from an image model does not retain/send an image to Audex.
- [ ] `sample_speech.wav` produces a visible coherent answer in Dinoki/Chat.
- [ ] `mlk_speech.wav` produces a visible coherent answer in Dinoki/Chat.
- [ ] A second conversational turn remains coherent.
- [ ] Every generation row records token/s.
- [ ] App-process `phys_footprint` stays within the accepted gate.
- [ ] Non-streaming `/v1/chat/completions` accepts `input_audio`.
- [ ] Streaming `/v1/chat/completions` accepts `input_audio`.
- [ ] Same audio/same prompt records expected cache lookup behavior.
- [ ] Different audio/same prompt cannot false-hit the same media key.
- [ ] Prefix/paged/L2 counters and disk-backed state are copied verbatim into
      the proof artifact.
- [ ] No output-audio controls are advertised.

### OpenAI-compatible payload

```json
{
  "model": "Nemotron-Labs-Audex-2B-4bit-vMLX",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_audio",
          "input_audio": {
            "data": "BASE64_WAV_BYTES",
            "format": "wav"
          }
        },
        {
          "type": "text",
          "text": "Transcribe the speech exactly."
        }
      ]
    }
  ],
  "stream": false,
  "temperature": 0
}
```

The response modality for this PR is text.

### Focused verification commands

vMLX focused tests:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test \
  --filter 'AudexTests|TokenizerAddedTokenRegexFocusedTests|audexModelTypeAdvertisesAudio' \
  --jobs 4
```

vMLX audio smoke:

```bash
GEMMA4_SMOKE_MODEL="$HOME/models/nvidia/Nemotron-Labs-Audex-2B-4bit-vMLX" \
GEMMA4_SMOKE_AUDIO=/path/to/speech.wav \
GEMMA4_SMOKE_PROMPT='Transcribe the speech exactly.' \
GEMMA4_SMOKE_MAX_TOKENS=128 \
swift run Gemma4AudioSmoke
```

Osaurus capability tests:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test \
  --package-path Packages/OsaurusCore \
  --filter 'd0_audexAudioOnly|audexModelTypeIsAudioOnlyWithoutVisionConfig|audexVLMFactoryIdentityDoesNotAdvertiseImages' \
  --jobs 2
```

The full Xcode developer directory is required on this machine because the
selected CommandLineTools SDK does not expose the Swift `Testing` framework.

### Review map

| Area | File |
| --- | --- |
| Architecture/processor | `Libraries/MLXVLM/Models/Audex.swift` |
| Factory dispatch | `Libraries/MLXVLM/VLMModelFactory.swift` |
| Capability snapshot | `Libraries/MLXLMCommon/ModelRuntimeCapabilitySnapshot.swift` |
| Tokenizer scaling | `Vendors/swift-transformers/Sources/Tokenizers/Tokenizer.swift` |
| Quantizer | `scripts/convert-audex-affine.py` |
| Runtime tests | `Tests/MLXLMTests/AudexTests.swift` |
| Full Osaurus contract | `docs/AUDEX-2B-OSAURUS-INTEGRATION-SPEC-2026-07-20.md` |
| Machine-readable contract | `docs/AUDEX-2B-OSAURUS-WIRING-MANIFEST.json` |

### Explicit non-goals

- causal speech decoder;
- XCodec/XCodec2 decode;
- text-to-audio CFG;
- enhancement VAE;
- audio response API/playback;
- hidden text-only masking of codec tokens;
- broad model-runtime cleanup.

### PR readiness statement

**PARTIAL — ready for vMLX source review; not yet ready for Osaurus production-consumption claims.**

Suggested wording before the Osaurus app proof:

> vMLX source and live audio-input correctness are ready for review. Osaurus
> consumption remains partial until the pushed vMLX SHA is pinned and the
> development app/API proof matrix passes. Output-audio modalities remain out
> of scope and must not be advertised.

---

## Short follow-up comment after Osaurus pinning

Use this only after filling every field from the actual pinned checkout:

> Osaurus integration refresh:
>
> - vMLX Audex commit: `REPLACE_WITH_PUSHED_AUDEX_VMLX_COMMIT`
> - Osaurus commit: `REPLACE_WITH_OSAURUS_COMMIT`
> - All four pin surfaces: `REPLACE_WITH_PIN_VERIFICATION`
> - Capability tests: `REPLACE_WITH_TEST_COUNT_AND_RESULT`
> - Dinoki audio rows: `REPLACE_WITH_ARTIFACT_PATH_AND_RESULT`
> - OpenAI `input_audio`: `REPLACE_WITH_ARTIFACT_PATH_AND_RESULT`
> - Cache topology/counters: `REPLACE_WITH_EXACT_TELEMETRY`
> - Peak physical footprint: `REPLACE_WITH_BYTES`
> - Token/s: `REPLACE_WITH_PER_ROW_VALUES`
>
> Text and audio-input -> text-output are promoted only if all named rows pass.
> Image, video, and output-audio remain hidden/unsupported.
