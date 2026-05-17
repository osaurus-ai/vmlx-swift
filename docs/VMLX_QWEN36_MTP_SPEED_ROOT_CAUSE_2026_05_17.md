# Qwen3.6 Native MTP Swift Speed Root Cause - 2026-05-17

This note records the focused Swift-side MTP speed investigation for
`/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP`.

## Result

The prior Swift default D3 MTP path was slower than autoregressive decode
because hybrid Qwen3.6 cache forced `NativeMTPTokenIterator` into
`sequential_repair`. That mode verifies one target token at a time for
hybrid/Mamba caches. It is correctness-first, but it removes the main D3
speedup: verifying `[primary, d1, d2, d3]` in one target forward.

Focused same-build evidence:

| Mode | Tok/s | Target forwards | Verify input tokens | Output | Artifact |
|---|---:|---:|---:|---|---|
| AR baseline | 31.7 | N/A | N/A | Coherent `1..50` count | `docs/local/qwen36-mtp-opt/20260517T045451Z-27b-mxfp4-verifier-mode/ar_baseline.log` |
| Default D3 | 27.1 | 188 | 188 | Coherent `1..50` count | `docs/local/qwen36-mtp-opt/20260517T045451Z-27b-mxfp4-verifier-mode/default_d3.log` |
| Chunk-commit D3 | 49.6 | 56 | 224 | Coherent `1..50` count | `docs/local/qwen36-mtp-opt/20260517T045451Z-27b-mxfp4-verifier-mode/chunk_commit_d3.log` |

The default D3 row is therefore a real slowdown. The chunk-commit row is the
real MTP speed path and clears the 44 tok/s target for this 27B MXFP4 artifact
on the focused text gate.

## Code Change

`Libraries/MLXLMCommon/SpecDec/NativeMTPTokenIterator.swift` now logs:

- `targetForwards`
- `verifyInputTokens`
- `repairForwards`
- `verifierMode`

It also accepts an explicit experimental switch:

```sh
VMLX_NATIVE_MTP_HYBRID_VERIFY=chunk_commit
```

Accepted values for the chunk path are `chunk`, `chunk_commit`,
`capture_commit`, and `fast`. Accepted values for the safe sequential path are
`sequential`, `sequential_repair`, and `repair`.

Default behavior remains conservative: hybrid/Mamba cache still uses
`sequential_repair` unless the chunk switch is explicitly set. Native MTP is
still not auto-enabled from a model name; it requires the real loaded MTP head
and explicit request path.

## Why It Was Slower

The default D3 row reported:

```text
verifierMode=sequential_repair targetForwards=188 verifyInputTokens=188
```

That means the runtime did roughly one target forward per generated token, then
paid the extra MTP draft overhead on top. It kept output coherent, but it could
not produce a speed win.

The chunk-commit row reported:

```text
verifierMode=chunk_commit targetForwards=56 verifyInputTokens=224
```

That is the expected D3 shape: many verifier cycles consume four verifier
positions at once. The model generated the same coherent count output and
reached 49.6 tok/s.

## Verification

Commands completed:

```sh
swift build -c release --product RunBench --jobs 2
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test --filter MTPRuntimeFocusedTests --jobs 2
```

Focused tests passed:

```text
28 tests in MTP runtime metadata passed
```

## Production Boundary

This is a focused text decode proof, not a full production release claim.
Before changing the default for Qwen3.6 hybrid MTP, run the six-artifact gate
with chunk-commit enabled:

- 27B JANG_4M
- 27B MXFP4
- 27B MXFP8
- 35B JANG_2K
- 35B MXFP4
- 35B MXFP8

Each artifact still needs:

- AR vs D1/D2/D3 speed and coherent output;
- cache off and cache on with prefix/paged/block-L2/SSM companion evidence;
- multi-turn text;
- reasoning on/off;
- VL image turn followed by text-only turn;
- RAM footprint;
- no loops, no hidden reasoning-only pass, and no forced sampling guard.

Only after those rows pass should `chunk_commit` become the default for the
Qwen3.6 hybrid MTP runtime.
