#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATED_DELTA="$ROOT/Libraries/MLXLLM/Models/GatedDelta.swift"
QWEN3NEXT="$ROOT/Libraries/MLXLLM/Models/Qwen3Next.swift"
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

require_text "$GATED_DELTA" 'Osaurus crash `APPLE-MACOS-54` reached `CustomKernel::eval_gpu`' \
  'Sentry crash rationale for guarding compute_g compile'
require_text "$GATED_DELTA" 'let body: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray' \
  'eager compute_g body'
require_text "$GATED_DELTA" 'let body: @Sendable ([MLXArray]) -> [MLXArray]' \
  'eager GatedDelta step-ops body'
require_text "$GATED_DELTA" 'return HardwareInfo.isCompiledDecodeSupported ? compile(shapeless: true, body) : body' \
  'HardwareInfo gate around GatedDelta compute_g compile'
require_text "$GATED_DELTA" 'return HardwareInfo.isCompiledDecodeSupported ? compile(body) : body' \
  'HardwareInfo gate around GatedDelta step-ops compile'
require_text "$GATED_DELTA" 'guard let kernel = selectedKernel, Dk >= 32, Dk % 32 == 0 else' \
  'pre-dispatch fallback when GatedDelta kernel is unavailable or tile shape is invalid'
require_text "$GATED_DELTA" 'return gatedDeltaOps(' \
  'ops fallback before GatedDelta Metal dispatch'
require_text "$QWEN3NEXT" 'Qwen3Next config linear_num_value_heads must be divisible by linear_num_key_heads.' \
  'decode-time SSM head divisibility validation'
require_text "$QWEN3NEXT" 'Qwen3Next config full_attention_interval must not exceed num_hidden_layers.' \
  'decode-time full-attention interval validation'
require_text "$QWEN3NEXT" 'Qwen3Next config num_experts_per_tok must not exceed num_experts.' \
  'decode-time MoE top-k validation'
reject_text "$GATED_DELTA" 'Always compiled, regardless of `HardwareInfo.isCompiledDecodeSupported`' \
  'old unconditional compile rationale'
reject_text "$QWEN3NEXT" 'precondition(numVHeads % numKHeads == 0' \
  'constructor head-divisibility precondition'
reject_text "$QWEN3NEXT" 'precondition(args.vocabularySize > 0' \
  'constructor vocabulary-size precondition'

if ! awk '
  /private let _compiledComputeG:/ { in_block=1 }
  in_block && /let body:/ { has_body=1 }
  in_block && /HardwareInfo\.isCompiledDecodeSupported \? compile\(shapeless: true, body\) : body/ { has_gate=1 }
  in_block && /^\}\(\)/ { in_block=0 }
  END { exit(has_body && has_gate ? 0 : 1) }
' "$GATED_DELTA"; then
  printf 'FAIL: %s _compiledComputeG is not structurally guarded by HardwareInfo\n' "$GATED_DELTA" >&2
  failures=$((failures + 1))
fi

if ! awk '
  /private let _compiledStepOps:/ { in_block=1 }
  in_block && /let body:/ { has_body=1 }
  in_block && /HardwareInfo\.isCompiledDecodeSupported \? compile\(body\) : body/ { has_gate=1 }
  in_block && /^\}\(\)/ { in_block=0 }
  END { exit(has_body && has_gate ? 0 : 1) }
' "$GATED_DELTA"; then
  printf 'FAIL: %s _compiledStepOps is not structurally guarded by HardwareInfo\n' "$GATED_DELTA" >&2
  failures=$((failures + 1))
fi

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Qwen3-Next/GatedDelta compute_g compile is guarded by HardwareInfo.\n'
