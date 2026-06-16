#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$ROOT/Libraries/MLXLLM/Models/MiniMaxJANGTQ.swift"

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

reject_text "$MODEL" '[MiniMaxJANGTQ sanitize] missing resident expert tensor' \
  'resident expert missing fatal text'
reject_text "$MODEL" 'fatalError(' \
  'process-fatal MiniMaxJANGTQ sanitize path'
reject_text "$MODEL" 'removeValue(' \
  'unchecked destructive tensor removal'

require_text "$MODEL" 'miniMaxJANGTQHasCompleteExpertSet(' \
  'complete expert set validation helper'
require_text "$MODEL" 'Leaving source q/k/v keys intact so load verification fails' \
  'QKV mismatch graceful load-verification path'
require_text "$MODEL" 'Leaving source expert keys intact so load verification fails' \
  'expert-set mismatch graceful load-verification path'

printf 'PASS: MiniMaxJANGTQ sanitize fails malformed layouts through load verification.\n'
