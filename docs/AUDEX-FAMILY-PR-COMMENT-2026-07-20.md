## Audex 2B + 30B-A3B quant/runtime handoff

This PR now covers both native model types:

- `nemotron_dense_audex` -> `Audex` (2B dense)
- `nemotron_h_audex` -> `AudexH` (30B MoE, 3B active)

Runtime PR: https://github.com/osaurus-ai/vmlx-swift/pull/168

Osaurus wiring PR: https://github.com/osaurus-ai/osaurus/pull/2116

### Quant repositories

- https://huggingface.co/OsaurusAI/Nemotron-Labs-Audex-2B-4bit
- https://huggingface.co/OsaurusAI/Nemotron-Labs-Audex-2B-6bit
- https://huggingface.co/OsaurusAI/Nemotron-Labs-Audex-2B-8bit
- https://huggingface.co/OsaurusAI/Nemotron-Labs-Audex-30B-A3B-4bit
- https://huggingface.co/OsaurusAI/Nemotron-Labs-Audex-30B-A3B-6bit
- https://huggingface.co/OsaurusAI/Nemotron-Labs-Audex-30B-A3B-8bit

The language decoder is MLX affine 4/6/8-bit, group size 64. NV-Whisper and
the audio projector remain source precision. The 30B converter pre-stacks all
46 routed-expert projection groups before quantization.

### Current proof

All six local bundles pass safetensors header/index validation and load through
`MLXLMCommon.loadModel`. In explicit NVIDIA instruct mode
(`enable_thinking=false`), every bundle completed a real audio transcription
plus two growing-history follow-ups through the production generation stream,
stopped normally, emitted visible coherent text, reported token/s on all three
turns, and leaked no protocol/reasoning markers.

| Bundle | Audio tok/s | Follow-ups tok/s | Peak sampled phys footprint |
| --- | ---: | ---: | ---: |
| 2B 4-bit | 117.1 | 119.6 / 114.7 | 6.427 GiB |
| 2B 6-bit | 117.6 | 114.9 / 113.8 | 6.413 GiB |
| 2B 8-bit | 121.8 | 114.8 / 122.1 | 6.897 GiB |
| 30B 4-bit | 50.8 | 50.6 / 51.3 | 24.127 GiB |
| 30B 6-bit | 50.0 | 51.1 / 49.8 | 31.452 GiB |
| 30B 8-bit | 49.5 | 50.2 / 50.2 | 38.944 GiB |

Required caveat: omitted/default thinking is not green. Representative 2B and
30B rows stopped with 195-196 reasoning characters and zero visible characters
because the model did not close the template's pre-opened reasoning envelope.
Do not add a forced close, hidden sampler override, or visible-output rewrite.
Dinoki/Osaurus should use the user's explicit instruct selection for the
proven audio path and keep default-thinking audio marked partial.

### Osaurus contract

- Detect names containing `nemotron-labs-audex` / `nemotron_labs_audex`.
- Detect config types `nemotron_dense_audex` and `nemotron_h_audex`.
- Advertise audio input only: no image/video fallback from VLM identity.
- Map OpenAI audio content to `Chat.Message.audios`; the processor owns sound
  placeholder expansion.
- Use batch size 1.
- Do not advertise TTS, text-to-audio, or speech-to-speech output from these
  input-audio quants.
- Do not claim prefix/paged/L2/TurboQuant-KV cache hits until counters and
  30B Mamba companion-state restore are live-proven.

Full human and machine-readable sheets:

- `docs/AUDEX-FAMILY-QUANT-RELEASE-2026-07-20.md`
- `docs/AUDEX-FAMILY-QUANT-RELEASE-2026-07-20.json`
