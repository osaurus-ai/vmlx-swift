# Ornith routed MTP verification follow-up

Status: draft; kernel improvement measured, whole-model numerical and app Adaptive qualification remain open. Builds on the Ornith loader PR #617. Ordinary prefill, AR, and draft-head dispatch are unchanged.

## Problem and change

Ornith JANGH2 selects eight routed experts per token. At verification width eight, the previous dispatch crossed the 64-routed-row prefill threshold, unlike single-token decoding. `Qwen35.nativeBackboneMTPVerifyForward` now establishes a task-local routed verification scope. Only matching single-batch target rows 2–8 with fewer than 64 routes per token request the existing mapped decode kernel. Flash verification retains its existing independent scope.

## Evidence and limits

On an M5 Max, real JANGH2 bundle, 38-token prompt and five interleaved repetitions, diagnostic target-forward timings were:

| Target step | Before ms | Candidate ms |
|---|---:|---:|
| AR | 24.31 | 24.05 |
| Verify 2 rows | 27.66 | 27.72 |
| Verify 4 rows | 35.04 | 35.04 |
| Verify 8 rows | 192.34 | 51.80 |

Eight-row graph construction dropped from 16.51 to 3.04 ms. These exclude drafter and app work, and are not end-to-end Adaptive throughput. All 48 checked target rows retained AR top-token agreement, but none was bitwise exact. Shared affine-expert and attention differences remain under investigation. No default/tuning artifact, sampler, seed, cache dtype, or adaptive policy is changed.

Two focused XCTest cases pass: the actual mixed 2/3-bit Hadamard BF16 mapped kernel matches serial decode at four/eight rows with eight routes; scope geometry, nesting, and throwing restoration are checked. The Release RunBench builds. `swift test -c release` encounters an unrelated existing DEBUG-only test API; Debug swiftbuild compilation succeeds but the discovery wrapper reports corrupted JSON. Direct execution of the built MLXLMTests.xctest runs the two named cases with zero failures.

Before merging/promoting: run the exact committed source in the Release app with matched Off/Adaptive long code/prose, verify the remaining numerical path, and retain context-growth, restore-dtype, SSD read/write and terminal-return measurements. The current app PR #3137 remains on the loader commit and does not yet consume this candidate.
