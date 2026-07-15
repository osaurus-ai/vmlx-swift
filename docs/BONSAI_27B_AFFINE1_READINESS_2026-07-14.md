# Bonsai 27B affine-1 readiness — 2026-07-14

## Scope

This checkpoint adds native Metal consumption of the schema-2 affine one-bit
weights in `/Users/eric/models/Bonsai-27b-1bit-JANG`. It also preserves the
bundle's exact per-tensor quantization manifest, including its four-bit vision
tensors, and hardens the real image/video/tool/reasoning proof harnesses.

The bundle declares Qwen 3.5 VLM image and video support. It declares
`audio_verified=false`; audio is therefore not a supported modality row for
this bundle and was not treated as a pass requirement.

## Source checkpoint

- vmlx-swift base: `a9b10f60e330337a9de2d8ebe3ca74a7370525e4`
- MLX dependency PR: `osaurus-ai/mlx#2`
- merged MLX integration pin: `9e01acd573a18540468160ccaffeb6fb566e891e`
- 1-bit bundle size: 4,472 MiB measured by the release harness
- schema-2 manifest: 581 entries
  - 498 language affine1/group-128 entries
  - 83 vision affine4/group-64 entries

The runtime keeps affine1 tensors packed. A lossless Swift-side expansion to
two bits was evaluated and rejected because it raised peak physical footprint
to about 8.8 GiB (202.5% of bundle size).

## Current source tests

All commands used a release build and the full Xcode toolchain.

| Gate | Result |
| --- | --- |
| Fresh local MLX Python build, `TestQuantized.test_affine_one_bit_metal` | PASS, 1 test |
| MLX pre-commit hooks on the six-file kernel change | PASS |
| `swift test -c release --filter JangAffine1RuntimeContractTests` | PASS, 6 tests |
| `swift test -c release --filter VLMProcessorCacheScopeSaltTests` | PASS, 7 tests |

The affine1 suite covers schema validation, malformed-contract rejection,
unaligned QMV, optimized 1024-wide QMV, QMM, and exact manifest resolution for
the ambiguous four-bit vision packing.

The complete repository suite is **not green** and is not claimed as evidence
for this PR. A default parallel run deadlocked in the existing opposite lock
order between `/vmlx_mlx_lock` and `MLXMetalTestLock`. An explicit
`--no-parallel` run avoided that deadlock but recorded failures in untouched
Gemma4 source-contract tests, emoji detokenizer tests, memory-safety-settings
expectations, DSV4 prompt tests, and other unrelated suites before the test
runner exited with signal 5. No failure was recorded in the changed affine1 or
Qwen3VL processor suites. The two focused suites above were rerun separately
from the current PR source and passed.

## 1-bit live release matrix

All rows below used the real bundle and production mmap-backed load policy.
No prompt coercion, forced reasoning closer, sampler clamp, hidden repetition
penalty, or synthetic generation default was added.

| Row | Live result | Verdict |
| --- | --- | --- |
| Text multi-turn | `SAVED` at 44.52 tok/s; callback `ORCHID` at 44.35 tok/s; clean stops | PASS |
| Text physical footprint | peak delta about 1.55 GiB, 35.7% of bundle size | PASS |
| Image, compile off | coherent synthetic red/blue gradient description at 39.8 tok/s; follow-up `blue` at 34.9 tok/s | PASS |
| Image, compile on | same grounded description at 39.2 tok/s; follow-up `blue` at 38.4 tok/s | PASS |
| Image physical footprint | peak delta 3,151 MiB, 70.4% of bundle size | PASS |
| Structured image cache | cold A; same-media disk restore HIT 99/99; coherent A replay; different-media MISS; grounded follow-up | PASS |
| Video multi-turn | real triangle fixture; grounded circle/triangle answer at 41.8 tok/s; foreground follow-up at 42.6 tok/s | PASS |
| Video physical footprint | peak delta 3,671 MiB, 82.1% of bundle size; post-turn footprint dropped to 1,765 MiB | PASS |
| Tool parser | structured `get_weather({"location":"Tokyo"})`; one tool event; no raw XML marker leakage | PASS |
| Reasoning parser | natural stop after 372 tokens at 39.2 tok/s; 332 reasoning deltas; visible `Answer: 4.`; closed reasoning; no markers | PASS |

The tool-only envelope measured 17.14 tok/s. That row validates structured
parser behavior, not the text throughput target; the release text rows are the
approximately 45 tok/s performance gate.

## Ternary regression matrix

`/Users/eric/models/Bonsai-27b-Ternary-JANG` remained coherent after the
affine1 work.

| Row | Live result | Verdict |
| --- | --- | --- |
| Text multi-turn | `SAVED` at 38.63 tok/s; `ORCHID` at 33.73 tok/s | PASS |
| Text physical footprint | peak delta about 1.37 GiB, 18.3% of bundle size | PASS |
| Image | grounded gradient and color follow-up, compile off/on; peak 34.6% of bundle size | PASS |
| Video | grounded circle/triangle outputs; 34.2 and 16.0 tok/s; peak 27.4% of bundle size | PASS |

The ternary row is a regression check, not a 45 tok/s claim.

## Rejected or superseded diagnostics

- Original vmlx main crashed the 1-bit bundle at the Metal bits assertion.
- The lossless Swift 1-to-2-bit expansion was coherent at 35.59 tok/s but
  failed low-RAM requirements at 202.5% of bundle size and was removed.
- The old plain-loader VL harness reached 145.6% of bundle size. It bypassed
  the production mmap policy; all VL harness entry points now use the
  production loader.
- Video initially peaked at 113.7% because completed media arrays remained in
  allocator working-set caches. Post-slot GPU fencing and media-only cache
  cleanup reduced the current row to 82.1%.
- A 128-token reasoning diagnostic stopped inside reasoning and failed by
  design. The 512-token row closed naturally and passed; no forced closer or
  length-cap pass was used.
- Qwen emitted valid tool XML before the processor fix, but `LMInput` had
  dropped the active schemas and the stream parser stripped the call. The
  processor now preserves schemas on both text and media inputs, and both the
  focused test and real structured tool row pass.

## Still pending after this runtime PR

These items must not be described as complete until their own evidence lands:

1. vmlx-swift PR CI and merge.
2. Focused Osaurus pin PR to the merged vmlx-swift revision.
3. Isolated signed Osaurus development build that does not replace or disturb
   the user's installed/running app.
4. Computer Use visual proof in that isolated build: model discovery/load,
   coherent text multi-turn, image/video where exposed by the app, tool and
   reasoning presentation, physical-footprint observation, and clean unload.
5. Finder AppleScript re-confirmation after the prior AppleEvents TCC reset.
6. Sustained repeated-spawn/delegation physical-footprint proof.
7. Separate automatic model routing and clearer hardware guidance work in the
   preserved Osaurus routing lane. That work is intentionally paused until the
   runtime and pin chain is merged.
