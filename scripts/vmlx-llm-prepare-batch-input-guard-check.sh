#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM="$ROOT/Libraries/MLXLLM/LLMModel.swift"
failures=0

require_text() {
  local file="$1"
  local pattern="$2"
  local label="$3"
  if ! grep -Fq "$pattern" "$file"; then
    printf 'FAIL: %s missing %s\n' "$file" "$label" >&2
    failures=$((failures + 1))
  fi
}

reject_text() {
  local file="$1"
  local pattern="$2"
  local label="$3"
  if grep -Fq "$pattern" "$file"; then
    printf 'FAIL: %s still contains forbidden %s\n' "$file" "$label" >&2
    failures=$((failures + 1))
  fi
}

reject_text "$LLM" 'fatalError(
                "LLMModel.prepare expects single-sequence input (batch=1)' \
  'LLMModel.prepare batch-input fatal'
require_text "$LLM" 'public enum LLMModelError: LocalizedError, Equatable' \
  'typed LLMModelError'
require_text "$LLM" 'case unsupportedBatchInput(shape: [Int])' \
  'unsupported batch input error case'
require_text "$LLM" 'throw LLMModelError.unsupportedBatchInput(shape: tokensShape)' \
  'throwing prepare batch-input refusal'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: LLMModel.prepare batch-input fatal boundary is throwing.\n'
