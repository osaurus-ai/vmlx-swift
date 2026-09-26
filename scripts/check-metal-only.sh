#!/usr/bin/env bash
# check-metal-only.sh: list every place in Libraries/ that assumes Metal without saying so.
#
# vmlx grew up on Apple silicon, so parts of it take "the GPU" to mean a Metal device. On a CPU,
# CUDA or Vulkan build those places fail at run time or pick the wrong path. Tag each one with a
# comment starting `METAL-ONLY:` that says which case applies, or guard the code with
# `#if canImport(Metal)`. The cases:
#
#   1. an "is GPU" test that guards a Metal kernel    (deviceType|defaultDevice()) ==/!= .gpu
#   2. a Metal kernel definition                      metalKernel(
#   3. an explicit GPU stream                         Stream.gpu
#
# For example, directly above a kernel that has no fallback:
#
#   // METAL-ONLY: case 2. Metal kernel with no fallback: this path needs a Metal device.
#
# A site passes when the text `METAL-ONLY:` appears within the five lines above it, so one tag
# covers any site in the next five lines; the convention is still one tag per site. `//` comment
# lines are skipped, and so is code inside an `#if canImport(Metal)` branch, which only
# Apple-platform builds compile. Exit status 1 when a site is untagged, 2 when there is no
# Libraries/ to scan.
#
#   scripts/check-metal-only.sh           # untagged sites and a summary
#   scripts/check-metal-only.sh --all     # also the tagged and guarded ones
#   scripts/check-metal-only-test.sh      # this script's tests
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 - "$@" <<'PY'
import os
import re
import sys

show_all = "--all" in sys.argv[1:]
CASES = [
    ("1", re.compile(r"(deviceType|defaultDevice\(\))\s*(==|!=)\s*\.gpu")),
    ("2", re.compile(r"metalKernel\(")),
    ("3", re.compile(r"Stream\.gpu\b")),
]
if not os.path.isdir("Libraries"):
    print(f"check-metal-only: no Libraries/ in {os.getcwd()}", file=sys.stderr)
    sys.exit(2)
untagged = tagged = guarded = 0
for directory, subdirectories, files in os.walk("Libraries"):
    subdirectories.sort()
    for name in sorted(files):
        if not name.endswith(".swift"):
            continue
        path = os.path.join(directory, name)
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        metal = []  # per open #if: whether its current branch is `canImport(Metal)`
        for number, raw in enumerate(lines, 1):
            line = raw.strip()
            if line.startswith("#if"):  # also C #ifdef/#ifndef in Metal source strings
                metal.append(re.fullmatch(r"#if\s+canImport\(Metal\)\s*(//.*)?", line) is not None)
                continue
            if line.startswith(("#elseif", "#else")):
                if metal:
                    metal[-1] = False
                continue
            if line.startswith("#endif"):
                if metal:
                    metal.pop()
                continue
            if line.startswith("//"):
                continue
            for case, pattern in CASES:
                if not pattern.search(line):
                    continue
                if any(metal):
                    guarded += 1
                    if show_all:
                        print(f"{path}:{number}: case {case}: guarded by canImport(Metal)")
                elif any("METAL-ONLY:" in above for above in lines[max(0, number - 6):number - 1]):
                    tagged += 1
                    if show_all:
                        print(f"{path}:{number}: case {case}: tagged")
                else:
                    untagged += 1
                    print(f"{path}:{number}: case {case}: untagged: {line[:100]}")
print(f"METAL-ONLY: {untagged} untagged, {tagged} tagged, {guarded} guarded by canImport(Metal)")
sys.exit(1 if untagged else 0)
PY
