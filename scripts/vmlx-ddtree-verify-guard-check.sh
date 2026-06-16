#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREE_VERIFY="$ROOT_DIR/Libraries/MLXLMCommon/SpecDec/TreeVerify.swift"
TREE_BUILDER="$ROOT_DIR/Libraries/MLXLMCommon/SpecDec/TreeBuilder.swift"

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

[[ -f "$TREE_VERIFY" ]] || fail "missing TreeVerify.swift"
[[ -f "$TREE_BUILDER" ]] || fail "missing TreeBuilder.swift"

if grep -Eq 'precondition\(|fatalError\(' "$TREE_VERIFY"; then
  fail 'TreeVerify must throw typed SpecDec errors instead of process-fatal validation'
fi

grep -q 'throw SpecDecError.invalidRequest("DDTree verify tree must have a root")' "$TREE_VERIFY" \
  || fail 'TreeVerify must reject empty compiled trees with a typed error'

grep -q 'DDTree verify parent chain broken' "$TREE_VERIFY" \
  || fail 'TreeVerify must reject broken parent chains with a typed error'

if grep -q 'precondition(!posteriorTokens.isEmpty' "$TREE_BUILDER"; then
  fail 'TreeBuilder.followVerifiedTree must not precondition-abort on empty posterior tokens'
fi

grep -q 'DDTree posteriorTokens must have at least one entry' "$TREE_BUILDER" \
  || fail 'followVerifiedTree must throw for empty posterior tokens'

grep -q 'DDTree child map index out of range' "$TREE_BUILDER" \
  || fail 'followVerifiedTree must throw for invalid child-map traversal'

printf 'PASS: DDTree verify/walk malformed state returns typed SpecDec errors.\n'
