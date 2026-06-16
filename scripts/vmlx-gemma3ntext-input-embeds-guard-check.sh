#!/usr/bin/env bash
set -euo pipefail

target="Libraries/MLXLLM/Models/Gemma3nText.swift"

if rg -n 'fatalError\("Cannot generate per layer inputs without input ids"\)' "$target" >/dev/null; then
  echo "FAIL: Gemma3nText inputsEmbeds path still aborts without token ids"
  exit 1
fi

if ! perl -0ne 'exit(/let perLayerInputsProcessed: MLXArray\?\s*if let perLayerInputs \{\s*perLayerInputsProcessed = perLayerInputs\s*\} else if let inputs \{\s*perLayerInputsProcessed = getPerLayerInputs\(inputs\)\s*\} else \{\s*perLayerInputsProcessed = nil\s*\}/s ? 0 : 1)' "$target"; then
  echo "FAIL: Gemma3nText inputsEmbeds path must use projection-only per-layer inputs"
  exit 1
fi

if ! rg -n 'func projectPerLayerInputs\(_ inputsEmbeds: MLXArray, perLayerInputs: MLXArray\?\) -> MLXArray' "$target" >/dev/null; then
  echo "FAIL: Gemma3nText projection-only helper contract is missing"
  exit 1
fi

echo "PASS: Gemma3nText inputsEmbeds path uses projection-only per-layer inputs instead of fatalError."
