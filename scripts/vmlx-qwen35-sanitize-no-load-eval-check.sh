#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM="$ROOT/Libraries/MLXLLM/Models/Qwen35.swift"
VLM="$ROOT/Libraries/MLXVLM/Models/Qwen35.swift"
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

reject_in_function() {
  local file="$1"
  local function_name="$2"
  local forbidden="$3"
  local label="$4"
  if awk -v fn="$function_name" -v bad="$forbidden" '
    index($0, fn) { in_fn=1; depth=0 }
    in_fn {
      depth += gsub(/\{/, "{")
      depth -= gsub(/\}/, "}")
      if (index($0, bad)) found=1
      if (depth == 0 && NR > 1) in_fn=0
    }
    END { exit(found ? 0 : 1) }
  ' "$file"; then
    printf 'FAIL: %s %s contains forbidden %s\n' "$file" "$function_name" "$label" >&2
    failures=$((failures + 1))
  fi
}

require_text "$LLM" 'normConvention: Self.normConvention(metadata)' \
  'metadata-driven Qwen35 LLM norm convention'
require_text "$VLM" 'normConvention: Self.normConvention(metadata)' \
  'metadata-driven Qwen35 VLM norm convention'
require_text "$LLM" 'Do not sample norm tensor values during sanitize' \
  'LLM source comment documenting no load-time value probe'
require_text "$VLM" 'Do not sample norm tensor values during sanitize' \
  'VLM source comment documenting no load-time value probe'
require_text "$LLM" 'Do not sample MTP norm tensor values during sanitize' \
  'LLM source comment documenting no MTP load-time value probe'
require_text "$VLM" 'Do not sample MTP norm tensor values during sanitize' \
  'VLM source comment documenting no MTP load-time value probe'
require_text "$LLM" 'Qwen35 config \(key.stringValue) must be greater than zero' \
  'LLM config decode rejects non-positive vocab before embedding construction'
require_text "$VLM" 'Qwen35 VLM config \(key.stringValue) must be greater than zero' \
  'VLM config decode rejects non-positive vocab before embedding construction'
require_text "$LLM" 'linear_num_value_heads must be divisible by linear_num_key_heads' \
  'LLM config decode rejects invalid GatedDelta head divisibility'
require_text "$VLM" 'linear_num_value_heads must be divisible by linear_num_key_heads' \
  'VLM config decode rejects invalid GatedDelta head divisibility'
require_text "$LLM" 'num_attention_heads must be divisible by num_key_value_heads' \
  'LLM config decode rejects invalid attention/KV head divisibility'
require_text "$VLM" 'num_attention_heads must be divisible by num_key_value_heads' \
  'VLM config decode rejects invalid attention/KV head divisibility'
require_text "$VLM" 'return HardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body' \
  'VLM shapeless compile helpers are gated by HardwareInfo'
require_text "$VLM" 'return HardwareInfo.isCompiledDecodeSupported ? compile(body) : body' \
  'VLM GatedDelta step-ops compile is gated by HardwareInfo'
require_text "$VLM" 'guard let kernel = selectedKernel, Dk >= 32, Dk % 32 == 0 else' \
  'VLM GatedDelta falls back before unavailable/tile-incompatible Metal dispatch'

reject_in_function "$LLM" 'private static func baseNormWeightsNeedShift' '.item(' \
  'load-time scalar eval'
reject_in_function "$VLM" 'private static func baseNormWeightsNeedShift' '.item(' \
  'load-time scalar eval'
reject_in_function "$LLM" 'private static func baseNormWeightsNeedShift' '.mean()' \
  'load-time mean eval'
reject_in_function "$VLM" 'private static func baseNormWeightsNeedShift' '.mean()' \
  'load-time mean eval'
reject_in_function "$LLM" 'private static func mtpNormWeightsNeedShift' '.item(' \
  'MTP load-time scalar eval'
reject_in_function "$VLM" 'private static func mtpNormWeightsNeedShift' '.item(' \
  'MTP load-time scalar eval'
reject_in_function "$LLM" 'private static func mtpNormWeightsNeedShift' '.mean()' \
  'MTP load-time mean eval'
reject_in_function "$VLM" 'private static func mtpNormWeightsNeedShift' '.mean()' \
  'MTP load-time mean eval'

if grep -Fq 'precondition(args.vocabularySize > 0)' "$LLM" "$VLM"; then
  printf 'FAIL: Qwen35 still relies on process-fatal vocabulary preconditions.\n' >&2
  failures=$((failures + 1))
fi

if grep -Fq 'numVHeads % numKHeads == 0' "$LLM" "$VLM"; then
  printf 'FAIL: Qwen35 still relies on process-fatal GatedDelta head divisibility preconditions.\n' >&2
  failures=$((failures + 1))
fi

if grep -Fq 'fatalError("VLM gated delta kernel not available")' "$VLM"; then
  printf 'FAIL: Qwen35 VLM still process-aborts when GatedDelta kernel is unavailable.\n' >&2
  failures=$((failures + 1))
fi

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Qwen35 base norm sanitize path does not force load-time MLX scalar eval.\n'
