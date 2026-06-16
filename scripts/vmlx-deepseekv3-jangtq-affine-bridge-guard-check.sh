#!/usr/bin/env bash
set -euo pipefail

model="Libraries/MLXLLM/Models/DeepseekV3JANGTQ.swift"
factory="Libraries/MLXLLM/LLMModelFactory.swift"

if rg -n 'fatalError\(\s*"DeepseekV3JANGTQConfiguration\.asAffine encode/decode failed:' "$model" >/dev/null; then
  echo "FAIL: DeepseekV3 JANGTQ affine bridge still process-aborts"
  exit 1
fi

if ! rg -n 'fileprivate func asAffine\(\) throws -> DeepseekV3Configuration' "$model" >/dev/null; then
  echo "FAIL: DeepseekV3 JANGTQ asAffine bridge must be throwing"
  exit 1
fi

if ! rg -n 'public init\(_ args: DeepseekV3JANGTQConfiguration\) throws' "$model" >/dev/null; then
  echo "FAIL: DeepseekV3JANGTQModel init must be throwing"
  exit 1
fi

if ! rg -n 'self\.affineConfig = try args\.asAffine\(\)' "$model" >/dev/null; then
  echo "FAIL: DeepseekV3JANGTQModel init must propagate asAffine errors"
  exit 1
fi

if ! rg -n 'return try DeepseekV3JANGTQModel\(config\)' "$factory" >/dev/null; then
  echo "FAIL: LLM factory must propagate DeepseekV3JANGTQModel construction errors"
  exit 1
fi

echo "PASS: DeepseekV3 JANGTQ affine bridge throws instead of process-aborting."
