#!/usr/bin/env bash
set -euo pipefail

media="Libraries/MLXVLM/MediaProcessing.swift"
errors="Libraries/MLXVLM/VLMModelFactory.swift"

if rg -n 'precondition\(videoFrames\.isEmpty == false\)' "$media" >/dev/null; then
  echo "FAIL: empty frame-backed video still process-aborts in MediaProcessing"
  exit 1
fi

if ! rg -n 'case emptyVideoFrames' "$errors" >/dev/null; then
  echo "FAIL: VLMError must expose a typed emptyVideoFrames error"
  exit 1
fi

if ! perl -0ne 'exit(/guard videoFrames\.isEmpty == false else \{\s*throw VLMError\.emptyVideoFrames\s*\}/s ? 0 : 1)' "$media"; then
  echo "FAIL: frame-backed video processing must throw emptyVideoFrames before sampling"
  exit 1
fi

echo "PASS: empty frame-backed video fails gracefully before media sampling."
