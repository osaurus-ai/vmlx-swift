#!/usr/bin/env bash
set -euo pipefail

file="Libraries/MLXLLM/Models/Qwen35JANGTQ.swift"
factory="Libraries/MLXLLM/LLMModelFactory.swift"

require_text() {
  local needle="$1"
  local target="${2:-$file}"
  if ! grep -Fq "$needle" "$target"; then
    echo "missing expected text: $needle" >&2
    exit 1
  fi
}

reject_text() {
  local needle="$1"
  local target="${2:-$file}"
  if grep -Fq "$needle" "$target"; then
    echo "found rejected text: $needle" >&2
    exit 1
  fi
}

require_text "fileprivate func asAffine() throws -> Qwen35TextConfiguration"
require_text "let affine = try args.asAffine()"
require_text "init(_ args: Qwen35JANGTQTextConfiguration, layerIdx: Int) throws"
require_text "public init(_ args: Qwen35JANGTQTextConfiguration) throws"
require_text "public init(_ args: Qwen35JANGTQConfiguration) throws"
require_text "return try Qwen35JANGTQModel(config)" "$factory"
reject_text "fatalError("

echo "Qwen35 JANGTQ affine bridge guard check passed"
