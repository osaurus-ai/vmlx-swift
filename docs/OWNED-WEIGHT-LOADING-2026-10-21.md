# Owned model-weight loading: review checkpoint

Status: implementation under validation; not ready to merge or release.

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

A growing Flint app turn rebuilt roughly 21k tokens after decoding to recreate a prompt-minus-one checkpoint, costing about 14 seconds. This is a separate boundary-storage defect, not solved by this loader change. Cache-store comments about shared copies also disagree with the current deep-copy helper and need a peak-memory audit. Real-app proof of the new loading path is pending.

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
