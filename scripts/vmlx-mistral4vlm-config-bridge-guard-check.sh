#!/usr/bin/env bash
set -euo pipefail

model="Libraries/MLXVLM/Models/Mistral4VLM.swift"
factory="Libraries/MLXVLM/VLMModelFactory.swift"

if rg -n 'try! JSONDecoder\(\)\.decode' "$model" >/dev/null; then
  echo "FAIL: Mistral4VLM projector config bridge still force-decodes"
  exit 1
fi

if ! rg -n 'public init\(_ config: Mistral4VLMConfiguration\) throws' "$model" >/dev/null; then
  echo "FAIL: Mistral4VLM init must be throwing so config bridge failures surface"
  exit 1
fi

if ! rg -n 'try Mistral4VLM\(config\)' "$factory" >/dev/null; then
  echo "FAIL: VLM factory must propagate Mistral4VLM construction errors"
  exit 1
fi

if ! perl -0ne 'exit(/static func projectorConfiguration\(from m4: Mistral4VLMConfiguration\) throws -> Mistral3VLMConfiguration.*try JSONDecoder\(\)\.decode/s ? 0 : 1)' "$model"; then
  echo "FAIL: Mistral4VLM projector config bridge must decode through a throwing helper"
  exit 1
fi

echo "PASS: Mistral4VLM projector config bridge throws instead of force-decoding."
