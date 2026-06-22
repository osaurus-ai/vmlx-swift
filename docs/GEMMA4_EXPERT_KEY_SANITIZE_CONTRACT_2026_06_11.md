# Gemma4 Expert Key Sanitizer Contract - 2026-06-11

## Regression

Gemma4 MoE source/BF16 bundles can store fused expert tensors as suffixless
keys:

- `model.language_model.layers.N.experts.gate_up_proj`
- `model.language_model.layers.N.experts.down_proj`

If those keys reach `TextExperts` unchanged, load fails with:

```text
Unhandled keys ["down_proj", "gate_up_proj"] in language_model.model.layers.0.experts in Gemma4.G4LanguageModel.TextModel.TextLayer.TextExperts
```

This is a loader/sanitizer regression, not a prompt, sampler, tool-call, or
cache-policy issue.

## Observed Layouts

`google--gemma-4-26B-A4B-it-qat-q4_0-unquantized` source/BF16 row:

- `model.language_model.layers.0.experts.down_proj`
  shape `[128, 2816, 704]`, dtype `BF16`
- `model.language_model.layers.0.experts.gate_up_proj`
  shape `[128, 1408, 2816]`, dtype `BF16`

`OsaurusAI/gemma-4-26B-A4B-it-qat-MXFP4` quantized row:

- `language_model.model.layers.0.experts.down_proj.weight`
- `language_model.model.layers.0.experts.down_proj.scales`
- `language_model.model.layers.0.experts.gate_up_proj.weight`
- `language_model.model.layers.0.experts.gate_up_proj.scales`

`OsaurusAI/gemma-4-26B-A4B-it-qat-JANG_4M` quantized row is already normalized
to the runtime module tree:

- `language_model.model.layers.0.experts.switch_glu.gate_proj.*`
- `language_model.model.layers.0.experts.switch_glu.up_proj.*`
- `language_model.model.layers.0.experts.switch_glu.down_proj.*`

E2B did not expose this because it has no MoE expert tensors.

## Required Normalization

The Gemma4 sanitizer must:

1. Strip a leading `model.` prefix.
2. Normalize `language_model.layers.` to `language_model.model.layers.`.
3. Rewrite suffixless and suffixed direct down projections:
   - `.experts.down_proj`
   - `.experts.down_proj.<suffix>`
   to:
   - `.experts.switch_glu.down_proj`
   - `.experts.switch_glu.down_proj.<suffix>`
4. Split suffixless and suffixed fused gate/up projections:
   - `.experts.gate_up_proj`
   - `.experts.gate_up_proj.<suffix>`
   into:
   - `.experts.switch_glu.gate_proj`
   - `.experts.switch_glu.up_proj`
   using `moe_intermediate_size` as the split point on axis 1.

## Regression Test

Run from `vmlx-swift`:

```bash
swift test --filter gemma4SanitizeSplitsFusedMoEExpertWeights
```

This test must include both source/BF16 suffixless keys and quantized suffixed
keys. Removing either layout from the test reopens this class of Gemma4 load
regression.
