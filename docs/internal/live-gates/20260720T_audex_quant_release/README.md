# Audex quant release live gate

Date: 2026-07-20

This directory records the local source/static/runtime gate used before the six
OsaurusAI Audex quant uploads.

Static reports:

- `2b-4bit-static.json`
- `2b-6bit-static.json`
- `2b-8bit-static.json`
- `30b-4bit-static.json`
- `30b-6bit-static.json`
- `30b-8bit-static.json`

The live matrix and required negative evidence are in
`live-runtime-results.json`. The passing command shape was:

```bash
GEMMA4_SMOKE_MODEL=/absolute/path/to/bundle \
GEMMA4_SMOKE_AUDIO=/Users/eric/models/nvidia/Nemotron-Labs-Audex-Space/examples/sample_speech.wav \
GEMMA4_SMOKE_PROMPT='Transcribe the speech accurately.' \
GEMMA4_SMOKE_MAX_TOKENS=96 \
GEMMA4_SMOKE_MULTI_TURN=1 \
GEMMA4_SMOKE_ENABLE_THINKING=false \
.build/arm64-apple-macosx/debug/Gemma4AudioSmoke
```

The executable used `MLXLMCommon.loadModel`, the bundle tokenizer/template and
generation defaults, real audio preprocessing, and the production
`MLXLMCommon.generate` event pipeline. It rejected empty visible output,
length stops, zero token/s, failed semantic recall, and visible control
markers. Physical footprint was sampled from `TASK_VM_INFO.phys_footprint`
every 100 ms.

The earlier direct `TokenIterator` rows were diagnostic only and are not used
as production-stream proof because they bypass reasoning/control-marker event
routing.
