#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FACTORY="$ROOT/Libraries/MLXVLM/VLMModelFactory.swift"
GEMMA4="$ROOT/Libraries/MLXVLM/Models/Gemma4.swift"
failures=0

require_text() {
  local file="$1"
  local pattern="$2"
  local label="$3"
  if ! grep -Fq "$pattern" "$file"; then
    printf 'FAIL: missing %s\n  pattern: %s\n  file: %s\n' "$label" "$pattern" "$file" >&2
    failures=$((failures + 1))
  fi
}

require_text "$FACTORY" '"gemma4_unified": create(Gemma4Configuration.self, Gemma4.init)' \
  'Gemma4 unified VLM model_type alias'
require_text "$FACTORY" '"gemma4_unified_vlm": create(Gemma4Configuration.self, Gemma4.init)' \
  'Gemma4 unified VLM explicit alias'
require_text "$FACTORY" '"Gemma4UnifiedProcessor": create(' \
  'Gemma4 unified processor alias'
require_text "$FACTORY" 'Gemma4ProcessorConfiguration.self, Gemma4Processor.init' \
  'Gemma4 unified alias uses real Gemma4 processor'
require_text "$GEMMA4" '?? visionConfig.defaultOutputLength' \
  'Gemma4 unified fallback for missing top-level vision soft token count'
require_text "$GEMMA4" 'vision_config.num_soft_tokens' \
  'Gemma4 unified shipped-config rationale'
require_text "$GEMMA4" '@ModuleInfo(key: "vision_embedder") private var visionEmbedder: UnifiedVisionEmbedder?' \
  'Gemma4 unified vision_embedder module'
require_text "$GEMMA4" '@ModuleInfo(key: "patch_dense") var patchDense: Linear' \
  'Gemma4 unified patch_dense weight consumer'
require_text "$GEMMA4" '@ModuleInfo(key: "pos_embedding") var posEmbedding: MLXArray' \
  'Gemma4 unified pos_embedding weight consumer'
require_text "$GEMMA4" 'positionIds: concatenated(prepared.map { $0.positionIds }, axis: 0)' \
  'Gemma4 unified processor carries image_position_ids'
require_text "$GEMMA4" 'counts: prepared.map { $0.count }' \
  'Gemma4 unified expands image placeholders by valid patch count'

if (( failures > 0 )); then
  exit 1
fi

printf 'PASS: Gemma4 unified shipped VLM aliases dispatch to Gemma4.\n'
