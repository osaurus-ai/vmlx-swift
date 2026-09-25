#!/usr/bin/env bash
# Tests for check-metal-only.sh, against a synthetic Libraries/ tree.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/scripts" "$work/Libraries/Demo"
cp "$here/check-metal-only.sh" "$work/scripts/"
cat > "$work/Libraries/Demo/Sites.swift" <<'SWIFT'
func b() {
    let k = MLXFast.metalKernel(name: "b")
}
func a() {
    // METAL-ONLY: case 2, no fallback.
    let k = MLXFast.metalKernel(name: "a")
}
#if canImport(Metal)
func c() { let k = MLXFast.metalKernel(name: "c") }
#else
func c() { if Device.defaultDevice().deviceType == .gpu { } }
#endif
// let k = MLXFast.metalKernel(name: "comment")
func d() {
    // METAL-ONLY: too far away
    //
    //
    //
    //
    //
    Stream.gpu.synchronize()
}
func e() {
    // METAL-ONLY: exactly five lines above
    //
    //
    //
    //
    Stream.gpu.synchronize()
}
#if !canImport(Metal)
func f() { let k = MLXFast.metalKernel(name: "f") }
#endif
#if canImport(Metal) // trailing comment
#if DEBUG
func g() { Stream.gpu.synchronize() }
#endif
#endif
func h() { if Device.defaultDevice() != .gpu { } }
#if canImport(Metal)
let header = """
    #ifndef TILE
    #endif
    """
func i() { let k = MLXFast.metalKernel(name: "i") }
#endif
SWIFT
printf 'metalKernel(\n' > "$work/Libraries/Demo/NOTES.md"
fail=0
set +e
out="$(bash "$work/scripts/check-metal-only.sh")"
status=$?
set -e
expected='Libraries/Demo/Sites.swift:2: case 2: untagged: let k = MLXFast.metalKernel(name: "b")
Libraries/Demo/Sites.swift:11: case 1: untagged: func c() { if Device.defaultDevice().deviceType == .gpu { } }
Libraries/Demo/Sites.swift:21: case 3: untagged: Stream.gpu.synchronize()
Libraries/Demo/Sites.swift:32: case 2: untagged: func f() { let k = MLXFast.metalKernel(name: "f") }
Libraries/Demo/Sites.swift:39: case 1: untagged: func h() { if Device.defaultDevice() != .gpu { } }
METAL-ONLY: 5 untagged, 2 tagged, 3 guarded by canImport(Metal)'
if [ "$out" != "$expected" ]; then
  echo "FAIL: output differs"; diff <(echo "$expected") <(echo "$out") || true; fail=1
fi
if [ "$status" != 1 ]; then echo "FAIL: exit status $status, expected 1"; fail=1; fi
printf 'func e() {\n    // METAL-ONLY: tagged\n    Stream.gpu.synchronize()\n}\n' > "$work/Libraries/Demo/Sites.swift"
if ! bash "$work/scripts/check-metal-only.sh" > /dev/null; then echo "FAIL: a clean tree must exit 0"; fail=1; fi
mkdir -p "$work/empty/scripts"
cp "$here/check-metal-only.sh" "$work/empty/scripts/"
set +e
bash "$work/empty/scripts/check-metal-only.sh" > /dev/null 2>&1
status=$?
set -e
if [ "$status" != 2 ]; then
  echo "FAIL: a tree without Libraries/ must exit 2, got $status"; fail=1
fi
[ "$fail" = 0 ] && echo "check-metal-only: all tests passed"
exit "$fail"
