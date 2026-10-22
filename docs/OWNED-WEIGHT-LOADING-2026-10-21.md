# Owned model-weight loading: review checkpoint

Status: owned-loading and reproduced MiniCPM loop fixes qualified by the bounded app matrix below. App integration still requires CI; no release is requested.

## Problem and change

The previous production loader could expose tensors as views into a mapped safetensors shard. Keeping any view alive could retain source storage after fused projections allocated replacement weights. Process footprint, allocator bytes and host wired memory consequently described different ownership and residency states. A low process value alone did not prove low total model residency.

This change uses allocator-owned packed model weights across the ordinary loader, JANGH/JANGHT/relU2 banks, mixed affine experts and DFlash companion loading. Darwin reads directly into the final allocation with uncached aligned I/O and page-sized edge scratch. It avoids a tensor-sized Foundation Data staging buffer. Excluded auxiliary tables remain excluded; Allosaurus n-gram tables remain SSD row reads.

Legacy mmap environment dispatch is removed in the MLX dependency. The async factory no longer mutates process-global mmap settings. Inherited Press streaming switches and the factory-selected JANGH mapping diagnostic cannot redirect production loading. Dormant low-level diagnostic APIs remain; this is not a claim that every historical Press source file was deleted.

Storage does not determine numerical precision. Existing exact affine metadata and raw JANGTQ tensors retain their established dtype policy. Seeds, sampling, speculative depth and compute kernels are unchanged. Safe Auto applies the existing bounded weight-plus-workspace scheduling allowance across families; explicit limits remain explicit.

## Verification gates

- Reader: scalar, empty and mixed dtypes; unaligned/page/chunk/EOF boundaries; cancellation; malformed shapes/spans; source mutation; legacy environment cannot enable mappings.
- Backing classification: actual MLX address and a containing file-backed region, not an unrelated later mapping returned by macOS. Check both owned storage and explicit mapped diagnostic controls.
- Runtime: actual Osaurus multi-turn runs, load/unload/reload, text/tools, adaptive/Off where supported, same-model and different-model delegation.
- Memory: separately record model bytes, process physical footprint, MLX active/reusable allocations, host wired/compressed/swap and SSD I/O. These overlap and must not be added together.
- Cache: active attention state and bounded I/O scratch require RAM. Historical checkpoints belong on SSD. Prove restore counts and prompt processing; inspect retained snapshots and writer payloads instead of inferring residency from a disk filename.
- Speed: load and first-token latency, prefill, sustained decode, post-answer/tool-dispatch delay. No all-family or speedup claim from loader tests.

## Open findings

A growing Flint app turn rebuilt roughly 21k tokens after decoding to recreate a prompt-minus-one checkpoint, costing about 14 seconds. This is a separate boundary-storage defect, not solved by this loader change. Cache-store comments about shared copies also disagree with the current deep-copy helper and need a peak-memory audit. The app matrix below covers owned loading; the broader checkpoint/writer findings remain separate follow-ups.

## Follow-up policy and telemetry corrections

App CI exposed a missing-host path: Safe Auto replaced the existing near-RAM
allowance with the base 70% limit when no Metal working-set ceiling was supplied.
The shared scheduling override now requires a measured positive ceiling. It
preserves the prior plan otherwise. `JANGHLoadPolicyTests` covers absent host,
absent ceiling and a known ceiling; all 11 tests pass through explicit XCTest.
The app's existing `bundleSpecificPlanReportsItsActualRaisedLoadBudget` remains
the independent consumer regression; app CI must pass after re-pinning.

The Allosaurus n-gram ready log incorrectly reported `warm=true` despite the
actual default-off branch. It now reports the decision actually used, including
F_NOCACHE suppression. No row-reader or warm-up policy changed. Updated live-log
confirmation is pending the rebuilt app.

On engine 1747b2bdb, actual app Auto/Off controls reached native MTP and AR.
Allosaurus code decode measured 39.5 vs 40.9 tok/s, respectively; prose measured
41.1 vs 40.0 tok/s, with substantially different reasoning lengths and a tool
call in the Off run. These single samples establish routing and disclose a
code loss; they do not establish an adaptive speedup or matched end-to-end gain.
Fused GDN and hyper projection routes remained active. Other speculative
families and concurrent/delegated requests are still under qualification.

Compatibility follow-up: 53 Swift Testing cases across
`VMLXMemorySafetySettingsTests`, `MemorySafetyLevelTests` and
`VMLXServerRuntimeSettingsTests` pass. They retain profile/default/custom caps,
Strict unknown/over-budget refusals, persisted cache opt-outs, sampler inheritance,
legacy policy migration and DFlash2 precedence/Off checks. Assertions that formerly
required mmap now require owned loading, the explicitly requested storage change;
Strict and custom limits were not relaxed. Local test signing initially failed
because an older diagnostic copied metallibs outside `Contents/Resources`; those
generated duplicates were moved out of the test bundle. Failures and final raw
results remain in the private evidence ledger. No product signing workaround was
added.

## Current app qualification and integration correction

Release app ed10b0af886a22adb9a771f8a9f274cd998a0000 pins engine
4561f961d4cdb2ba10b9dfd3251e2dd26c171489 and core
169ba206ac423da4a2fa9cba2e2c036430f3c2ce. Core PR #27 has merged.

Owned allocations exposed a previously ineligible lane-tiling path. Qwen shared
expert/router fused readers read raw affine weights; tiling these matrices
corrupted MiniCPM Auto output into repeated numbers. Their layouts now remain
ordinary affine, alongside the existing raw linear-attention reader exclusions.
Lane dispatch, other eligible tiling, fused compute and adaptive widths remain
active. A numerical raw-reader test fails before the fix and passes afterward.
A separate verifier correction requires explicit opt-in before compiled staged
verification; default eager staged verification remains enabled. That correction
alone did not fix looping, as recorded in the failed controls.

Actual Osaurus MiniCPM 35B JANG_4M checks on the final code:

| App action | Result | Rate | Process peak, decimal GB |
| --- | --- | --- | --- |
| Auto bicycle explanation | 1,043 tokens, normal stop, no loop | 64.68 tok/s | 28.34 |
| Connected Auto explanation + calculate | Tool returns 391; final answer 391 | 68.72 / 78.15 tok/s | 28.38 |
| UI Off bicycle control | 985 tokens, normal stop, no loop | 62.67 tok/s | 28.96 |

The follow-up restored 5,296/7,584 then 7,583/7,859 tokens from SSD.
Checkpoint writes of 173–225 MB completed in 52–60ms. Activity Monitor showed
23.89 GB for the actual Osaurus PID; packed base weights are 24.14 decimal GB
plus a 0.533 GB drafter. Live KV/workspace/app allocations remain necessary.
These single sampled rows establish routing and coherency, not a general speedup.

Earlier bounded owned-loader app rows also cover Ornith affine and JANGH2,
Allosaurus JANGHT, load/unload/reload, tools and SSD restore. Inspected Activity
Monitor readings were model-scale (Ornith affine 70.20 GB settled, H2 41.88 GB,
Allosaurus 28.37 GB). MiniCPM Strict/Keep-in-memory Off controls preserved the
explicit cap and disabled pinning. The earlier Allosaurus code Auto row lost
about 3.4% versus Off; it remains disclosed above.

Evidence: `~/vmlx-private-evidence/resident-weight-ownership-20261021/`, especially
`MINICPM-FINAL-APP-PROOF.md`, `ACTIVITY-MONITOR-RAM.md`, `SETTINGS-COMPATIBILITY.md`,
and `turns/S17-auto-layout-fixed`, `S18-auto-layout-followup`, `S19-off-layout-control`.
No diagnostic lane/staging overrides were set in the final app run. No seeds or
samplers were changed. Missing coverage remains explicit: calibrated Flint app
runs, broad concurrent/delegated family matrix, and long-context/media memory
qualification are separate work, not implied by these bounded receipts.
