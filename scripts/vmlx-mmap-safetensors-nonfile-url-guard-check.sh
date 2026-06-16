#!/usr/bin/env bash
set -euo pipefail

file="Libraries/MLXLMCommon/MmapSafetensorsLoader.swift"

require_text() {
  local needle="$1"
  if ! grep -Fq "$needle" "$file"; then
    echo "missing expected text: $needle" >&2
    exit 1
  fi
}

reject_text() {
  local needle="$1"
  if grep -Fq "$needle" "$file"; then
    echo "found rejected text: $needle" >&2
    exit 1
  fi
}

require_text "case nonFileURL(URL)"
require_text "guard url.isFileURL else {"
require_text "throw MmapSafetensorsError.nonFileURL(url)"
reject_text "precondition(url.isFileURL)"

echo "MmapSafetensorsLoader non-file URL guard check passed"
