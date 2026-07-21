# Audex-2B vMLX live-gate evidence — 2026-07-20

This directory preserves the raw evidence that supports the measured
quantized-audio row in the Audex runtime checkpoint. It is intentionally
separate from the later Osaurus/Dinoki application gate.

## Artifact identity

- Checkout base: `2bfa9caffb951b14a76c869353abe1ff230668bd`
- Model: `~/models/nvidia/Nemotron-Labs-Audex-2B-4bit-vMLX`
- Quantization: MLX affine 4-bit, group size 64
- Audio: `~/models/nvidia/Nemotron-Labs-Audex-Space/examples/mlk_speech.wav`
- Date/time of footprint capture: `2026-07-20 19:55` America/Los_Angeles
- Measured process: `Gemma4AudioSmoke` PID 16184

`Gemma4AudioSmoke` is a historical executable name. It calls the generic vMLX
loader and processor; the raw log identifies the loaded runtime model as
`Audex`.

## Preserved raw files

- `quantized_mlk_audio_smoke.log`: load, preparation, generation, token/s,
  visible response, and PASS marker.
- `quantized_mlk_phys_footprint.txt`: `/usr/bin/footprint` sampling output for
  the same process.

## Measured result

| Field | Value |
| --- | --- |
| Load | 4.5 s |
| Processor preparation | 48 ms |
| Prompt tokens | 776 |
| Normalized waveform | `[1, 480000]` |
| Generated tokens | 33 |
| Decode | 0.8 s, 43.2 tok/s |
| Peak physical footprint | 3,073,313,624 bytes |
| Visible result | English identified; coherent “I have a dream…” transcription |
| Protocol leakage | None observed |

The footprint row is physical-footprint telemetry; it was not created by
lowering `MLX.Memory.memoryLimit`.

## Evidence boundary

This artifact proves one direct vMLX quantized audio-input/text-output row. It
does not prove:

- the Osaurus or Dinoki UI path;
- the OpenAI-compatible `input_audio` path;
- streaming behavior;
- a second audio turn or audio prefix-cache hit;
- paged or L2 disk cache counters;
- output audio.

Those remain explicit promotion gates in the integration specification.
