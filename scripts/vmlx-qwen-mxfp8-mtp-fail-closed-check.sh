#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mtp="$root/Libraries/MLXLMCommon/SpecDec/MTPRuntime.swift"
specdec="$root/Libraries/MLXLMCommon/SpecDec/SpecDecRuntime.swift"
settings="$root/Libraries/MLXLMCommon/ServerRuntimeSettings.swift"
factory="$root/Libraries/MLXLLM/LLMModelFactory.swift"

fail=0

require() {
  local file="$1"
  local pattern="$2"
  local message="$3"
  if ! rg -q "$pattern" "$file"; then
    printf 'FAIL: %s\n  missing pattern: %s\n  file: %s\n' "$message" "$pattern" "$file" >&2
    fail=1
  fi
}

require "$mtp" 'public static let fileName = "vmlx_mtp_tuning\.json"' \
  'native MTP production tuning must be bundle-local and explicitly named.'
require "$mtp" 'nativeMTPTuning\?\.usableBestDepth != nil' \
  'MTP status must require usable tuning before production launch.'
require "$mtp" 'public var requiresNativeMTPTuningBeforeAutoLaunch' \
  'complete preserved MTP artifacts without tuning must be distinguishable from launchable MTP.'
require "$mtp" 'public var canAutoLaunchMTP' \
  'MTP status must expose a single auto-launch gate.'
require "$mtp" 'guard status\?\.canAutoLaunchMTP == true else' \
  'explicit native MTP load requests must fail closed without usable tuning.'
require "$mtp" 'throw NativeMTPActivationError\.requestedWithoutUsableTuning' \
  'unverified native MTP requests must surface a typed activation error.'
require "$mtp" 'guard !requireVerifiedRuntime \|\| status\.canAutoLaunchMTP else \{ return nil \}' \
  'MTP auto-decode policy must reject unverified artifacts before recommending a depth.'
require "$mtp" 'without `vmlx_mtp_tuning\.json` is loadable, but it does not receive' \
  'MTP policy must document that tensor-preserved artifacts without tuning stay autoregressive.'

require "$settings" 'resolvedMTPLaunch' \
  'server settings must resolve MTP launch from full bundle evidence.'
require "$settings" 'resolved\.nativeMTP = resolvedMTPLaunch' \
  'load configuration must preserve MTP tensors only for speculative launch.'
require "$settings" 'launch\.launchMode == \.speculative' \
  'draft strategy must exist only for a speculative launch recommendation.'
require "$settings" 'MTP cannot be forced on until the bundle has complete tensor evidence and usable vmlx_mtp_tuning\.json' \
  'force-on UI/settings validation must explain missing production tuning.'

require "$factory" 'NativeMTPActivation\.shouldLoadNativeMTPWeights' \
  'LLM factory must use the native MTP activation gate before model construction.'
require "$factory" 'NativeMTPActivation\.scrubInactiveMTPConfig' \
  'LLM factory must scrub inactive MTP config before normal autoregressive construction.'
require "$specdec" 'case invalidRequest\(String\)' \
  'spec-dec runtime must expose typed invalid-request errors instead of process-fatal validation.'
require "$specdec" 'throw SpecDecError\.invalidRequest\("DFlash block_size must be >= 2"\)' \
  'DFlash invalid block size must throw instead of precondition-aborting.'
require "$specdec" 'throw SpecDecError\.invalidRequest\("DDTree branchingBudget must be >= 1"\)' \
  'DDTree invalid branching budget must throw instead of precondition-aborting.'
require "$specdec" 'throw SpecDecError\.invalidRequest\("unexpected logits shape: ndim=' \
  'SpecDec unexpected logits rank must throw instead of fatalError.'

if rg -n 'precondition\(|fatalError\(' "$specdec" >/dev/null; then
  printf 'FAIL: SpecDecRuntime still contains process-fatal validation.\n' >&2
  rg -n 'precondition\(|fatalError\(' "$specdec" >&2
  fail=1
fi

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

printf 'PASS: Qwen MXFP8 native-MTP artifacts fail closed without usable bundle-local tuning.\n'
