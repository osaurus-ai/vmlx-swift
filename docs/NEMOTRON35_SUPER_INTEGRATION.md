# Nemotron 3.5 Super-VL integration checkpoint

Status: partial; engine #609 and app #3114 remain drafts. No release/tag.

## Implemented
- Read mixed affine projector quantization and physically validate final vision LayerNorm tensors independently of speculation settings.
- Load image-only Omni bundles without constructing Parakeet audio modules.
- Match image template expansion, rectangular pixel shuffle and float antialiased RGB preprocessing.
- Compute router logits and weighted expert sums in float32; retain float32 recurrent state and bound the SSM scan to 256-row chunks.
- Apply causal masking to cacheless language forwards.
- Use post-normalization states in the native MTP head and stage recurrent verify state for accepted-prefix commits.
- Preserve activation-dtype rounding in fused decode convolution, bias and SiLU. The fused path stays enabled.
- Keep graph construction and background prefetch evaluation on the same explicit MLX stream. This fixes a reproduced native-MTP worker-thread stream crash.
- Apply the vision projector RMS affine weight in FP32 before restoring activation dtype, matching the bundle rather than the generic fused RMSNorm rounding boundary.
- Restore interpolated RADIO position embeddings to patch activation dtype before addition, as required by the source bundle. Interpolation remains FP32.

## Current evidence
- Rectangular/square image shuffle: 2 tests; independent PyTorch image-resize vectors: 2 tests; FP32 CPE interpolation: 1 test; physical final-norm tensor checks: 4 tests, including stale indexes and incomplete files.
- Native staged-Mamba, normalization, fused-convolution rounding and cross-thread prefetch: 4/4 tests. Native MTP depth policy: 7/7 tests.
- Matched TF32 settings give exact teacher-forced agreement for all three text cases (77, 49 and 301 positions) and matching independent greedy outputs. No production precision policy changed.
- Independent image parity remains under investigation; do not call the raw default-backend teacher row passing. The supplied Python reference used the wrong vision block LayerNorm epsilon and omitted the source's post-interpolation dtype cast. The original fixture is preserved; separate corrected-reference receipts document both differences. Matched-input language-path isolation passes 349/349 positions with KL 1.04e-13 and all 23 greedy tokens through EOS. This is not independent image acceptance. A separate control using the same compiled Metal library makes all 32 RADIO blocks bit-exact; the default Python and Swift Metal builds first differ at five bf16 elements in block-0 LayerNorm. No production Metal library or precision policy is replaced.
- Native MTP real-bundle diagnostic: explicit depth 2, two capped easy-code rows, 36.98/41.97 tok/s versus AR 26.66/28.74; 124 prefetched verify cycles consumed in each row. These are diagnostic rows, not app acceptance, sustained performance or matched-clock speed claims.

## Live Osaurus evidence
App fdc7f95d8, engine 11df5292e9d3e51e52fcca00cebef13fb35d0d33:
- Connected calculator turns, motorcycle recognition and image-history follow-up produced coherent visible answers with separate reasoning and structured tools.
- Restart and tool-step disk restores include 8 KV + 40 Mamba layers. Approximately 212 MB tool snapshots take 21–22 ms for background saves and about 7 ms on the reply save path.
- Observed decode 33.8–35.8 tok/s; peak physical footprint approximately 7.1 GiB. CPU compilation overlapped: functional measurements, not controlled speed comparisons.
- On (Adaptive) resolves to AR because this bundle lacks usable tuning. Do not label those rows MTP.
- Direct Stop in a delegated child chat incorrectly returned partial work as success. App commit 0f62d0332 shares cancellation cleanup and preserves the cancellation result; 6/6 observer tests pass. Live Debug app after-check passes: child Stop returns ok:false/retryable:false, and a subsequent delegated calculator task completes with 20413. Raw records and public proof comment are on #3114.

## Remaining gates
1. Independent image comparison after the vendor dtype correction; focused regression for patch dtype.
2. Re-pin all six app sites to the final engine commit; rebuild and repeat affected image/delegation rows.
3. Native MTP sampled, media, cache and connected app checks before assigning measured defaults. No fabricated tuning, sampler changes or bundle writes.
4. Long-context resource/cache hardening, cancellation during model load and concurrent admission remain unproven for this family.
5. App CI and public source/live receipts before merge.

## Build notes
Use Swift's swiftbuild backend consistently. Engine tests use Release with DEBUG test helpers; app unit tests use the repository's Debug OsaurusCoreTests scheme. Release app builds are for live performance. A build failure before test execution is not a test failure or a pass.

Original Claude worktrees and Python reference source remain untouched. Temporary layer tracing is removed from deliverable source. Full diagnostic receipts, failed controls and source patches are retained privately; public PR comments summarize their provenance and limits.
