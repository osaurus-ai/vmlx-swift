#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$ROOT/Libraries/MLXLLM/Models/Laguna.swift"

require_text() {
  local file="$1"
  local text="$2"
  local label="$3"
  if ! grep -Fq "$text" "$file"; then
    printf 'FAIL: %s missing %s\n' "$file" "$label" >&2
    exit 1
  fi
}

reject_text() {
  local file="$1"
  local text="$2"
  local label="$3"
  if grep -Fq "$text" "$file"; then
    printf 'FAIL: %s still contains forbidden %s\n' "$file" "$label" >&2
    exit 1
  fi
}

reject_text "$MODEL" '[Laguna sanitize] layer' \
  'process-fatal Laguna sanitize diagnostic'
reject_text "$MODEL" 'QKV fusion requires identical' \
  'old Laguna QKV fatal message'
reject_text "$MODEL" 'out.removeValue(forKey: "\(qKey)' \
  'destructive q projection removal'
reject_text "$MODEL" 'out.removeValue(forKey: "\(kKey)' \
  'destructive k projection removal'
reject_text "$MODEL" 'out.removeValue(forKey: "\(vKey)' \
  'destructive v projection removal'

require_text "$MODEL" 'Leaving source q/k/v keys intact' \
  'Laguna QKV mismatch source-key preservation'
require_text "$MODEL" 'so load verification fails' \
  'Laguna QKV mismatch load-verification path'

printf 'PASS: Laguna sanitize fails malformed q/k/v layouts through load verification.\n'
