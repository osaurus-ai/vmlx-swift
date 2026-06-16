#!/usr/bin/env bash
set -euo pipefail

target="Libraries/MLXLLM/Lora+Data.swift"

if rg -n 'fatalError\("Unable to load data file, unknown type:' "$target" >/dev/null; then
  echo "FAIL: unsupported LoRA data file type still process-aborts"
  exit 1
fi

if ! rg -n 'case unsupportedFileType\(URL\)' "$target" >/dev/null; then
  echo "FAIL: LoRADataError must expose unsupportedFileType"
  exit 1
fi

if ! rg -n 'throw LoRADataError\.unsupportedFileType\(url\)' "$target" >/dev/null; then
  echo "FAIL: loadLoRAData(url:) must throw unsupportedFileType for unknown extensions"
  exit 1
fi

echo "PASS: unsupported LoRA data file types throw instead of process-aborting."
