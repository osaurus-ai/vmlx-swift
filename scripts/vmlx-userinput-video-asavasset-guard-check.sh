#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_INPUT="$ROOT_DIR/Libraries/MLXLMCommon/UserInput.swift"

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

[[ -f "$USER_INPUT" ]] || fail "missing UserInput.swift"

if grep -q 'calling asAVAsset() on Video Input with VideoFames provided is unsupported' "$USER_INPUT"; then
  fail 'UserInput.Video.frames asAVAsset path must not process-abort'
fi

grep -q 'public func asAVAsset() throws -> AVAsset' "$USER_INPUT" \
  || fail 'deprecated asAVAsset must be throwing so unsupported frames can fail gracefully'

grep -q 'case unsupportedVideoFramesAsAVAsset' "$USER_INPUT" \
  || fail 'UserInputError must include unsupported video frames asAVAsset error'

grep -q 'throw UserInputError.unsupportedVideoFramesAsAVAsset' "$USER_INPUT" \
  || fail 'Video.frames asAVAsset path must throw typed UserInputError'

printf 'PASS: UserInput.Video.frames asAVAsset fails gracefully instead of fatalError.\n'
