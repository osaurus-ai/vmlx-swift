#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIDDEN_CAPTURE="$ROOT_DIR/Libraries/MLXLMCommon/SpecDec/HiddenStateCapture.swift"
SPECDEC_RUNTIME="$ROOT_DIR/Libraries/MLXLMCommon/SpecDec/SpecDecRuntime.swift"
DFLASH_MODEL="$ROOT_DIR/Libraries/MLXLMCommon/SpecDec/DFlashDraftModel.swift"

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

[[ -f "$HIDDEN_CAPTURE" ]] || fail "missing HiddenStateCapture.swift"
[[ -f "$SPECDEC_RUNTIME" ]] || fail "missing SpecDecRuntime.swift"
[[ -f "$DFLASH_MODEL" ]] || fail "missing DFlashDraftModel.swift"

if grep -Eq 'precondition\(|fatalError\(' "$HIDDEN_CAPTURE"; then
  fail 'HiddenStateCapture must throw typed SpecDec errors instead of process-fatal validation'
fi

grep -q 'public func extractContextFeature' "$HIDDEN_CAPTURE" \
  || fail 'extractContextFeature helper is missing'

grep -q ') throws -> MLXArray' "$HIDDEN_CAPTURE" \
  || fail 'extractContextFeature must be throwing'

grep -q 'throw SpecDecError.invalidRequest("DFlash target_layer_ids must be non-empty")' "$HIDDEN_CAPTURE" \
  || fail 'empty DFlash target_layer_ids must throw'

grep -q 'DFlash missing captured hidden state for target layer' "$HIDDEN_CAPTURE" \
  || fail 'missing captured target hidden states must throw'

grep -q 'try extractContextFeature' "$SPECDEC_RUNTIME" \
  || fail 'SpecDec runtime must propagate extractContextFeature errors'

grep -q 'DFlash dflash_config.target_layer_ids must be non-empty' "$DFLASH_MODEL" \
  || fail 'DFlash config decode must reject empty target_layer_ids before drafter construction'

grep -q 'DFlash dflash_config.target_layer_ids must be nonnegative' "$DFLASH_MODEL" \
  || fail 'DFlash config decode must reject negative target_layer_ids'

printf 'PASS: DFlash hidden-state capture mismatches throw typed SpecDec errors.\n'
