# Nemotron 3.5 Super-VL integration checkpoint

Status: initial Super-VL AR text/image integration is locally verified; engine #609 and app #3114 await merge and final app CI. Native MTP remains explicitly partial. No release/tag.

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
- Independent image parity passes with the normal Swift Metal library: 349/349 teacher positions, KL 5.394e-14, matching vision checksums and all 23 greedy tokens through EOS. Two reference-helper deviations (vision block epsilon and post-interpolation dtype cast) were corrected in a separate fixture from the vendor source; the original fixture remains untouched. RADIO now explicitly fuses variance accumulation and applies its rounded normalization's affine weight in FP32, eliminating toolchain-dependent LayerNorm differences without changing shared MLX kernels.
- Vision projector regressions pass 2/2; patch interpolation/dtype regressions pass 2/2. The RADIO normalization regression passes across FP32/FP16/BF16, strided rows and a partial SIMD group.
- Native MTP real-bundle diagnostic: explicit depth 2, two capped easy-code rows, 36.98/41.97 tok/s versus AR 26.66/28.74; 124 prefetched verify cycles consumed in each row. These are diagnostic rows, not app acceptance, sustained performance or matched-clock speed claims.

## Live Osaurus evidence
App fdc7f95d8, engine 11df5292e9d3e51e52fcca00cebef13fb35d0d33:
- Connected calculator turns, motorcycle recognition and image-history follow-up produced coherent visible answers with separate reasoning and structured tools.
- Restart and tool-step disk restores include 8 KV + 40 Mamba layers. Approximately 212 MB tool snapshots take 21–22 ms for background saves and about 7 ms on the reply save path.
- Observed decode 33.8–35.8 tok/s; peak physical footprint approximately 7.1 GiB. CPU compilation overlapped: functional measurements, not controlled speed comparisons.
- On (Adaptive) resolves to AR because this bundle lacks usable tuning. Do not label those rows MTP.
- Direct Stop in a delegated child chat incorrectly returned partial work as success. App commit 0f62d0332 shares cancellation cleanup and preserves the cancellation result; 6/6 observer tests pass. Live Debug app after-check passes: child Stop returns ok:false/retryable:false, and a subsequent delegated calculator task completes with 20413. Raw records and public proof comment are on #3114.

## Updated Release app evidence
App 490718983 / engine 354d334ac, normal packaged Metal library:
- Four connected image turns distinguish NEMO427/red triangle/blue circle from NEMO853/green triangle/orange circle, recall the earlier image and compare both. All end at EOS. Decode 36.17–36.24 tok/s; first/new-image TTFT 8.57/9.53 s, cached follow-ups 1.50/1.79 s. The changed media salt prevents stale-image reuse.
- Five connected archive/code/prose turns grow actual prompt tokens through 45,813, 87,595, 129,805, 132,138 and 133,707. Recall answers are correct, and the final prose accurately describes the first and latest records. No loop, reasoning leakage or length stop.
- At 132,138 tokens, a 1,355-token code answer runs at 30.62 tok/s; the 650-token prose follow-up runs at 30.74 tok/s. Decode clocks span 1,424–1,615 MHz and 1,429–1,614 MHz respectively. These are AR results, not speculative-decoding or theoretical maximum claims.
- Every long follow-up restores all 48 layers from SSD. The code turn restores 129,799 tokens and processes 2,339; prose restores 132,132 and processes 1,575. TTFT is 8.96/8.28 s, including cache IO and preparation; it is not pure prefill throughput.
- Long-context background snapshots grow from 545 MB/169 ms to 1,266 MB/437 ms. Final two background writes take 542/437 ms; reply-path enqueue is separate. Peak process physical footprint is 8.09 GiB across the full session. Final isolated cache directory is 12.99 GB across several chats; no full-quota eviction claim.
- The isolated context cap and KV retention were explicitly set to 196,608 for this test. Safe Auto remained enabled, paged RAM cache stayed off and the normal idle-unload setting was unchanged. Default context policy was not modified.
- Actual app controls and visuals were inspected. The source/raw-turn/resource/cache receipt is recorded in #3114's proof comments. The code answer is a coherence/completion check, not a general coding-quality benchmark.

## Remaining merge gates
1. Pin the engine merge commit at all six app sites and pass required Osaurus CI.

Final app 5aa06b066 / engine 0ace9b23 passes the affected live progress/cancellation check: actual prefill advances 7,333 → 8,357 → 9,893 of 20,298 tokens; Stop records cancellation before output. The next request restores the earlier valid 4,773-token SSD prefix, calls the calculator and visibly answers 19 × 23 = 437. Cancelled partial state is not reused. Tool/final decode is 19.54/25.46 tok/s on very short rows with unrelated CPU compilation present; these are functional recovery checks, not performance comparisons. Model math and cache serialization are unchanged from the preceding 133k Release proof.

## Explicit follow-ups, not initial AR support claims
The supplied integration handoff treats native MTP as optional for initial support. Automatic MTP tuning, media MTP and connected app MTP are not validated; absent measured tuning the app uses AR. No invented tuning, sampler changes or bundle writes.

A real-bundle explicit-depth-2 MTP diagnostic with bundle sampling completed two turns, restored 8 KV + 40 Mamba layers and consumed 21/27 verify cycles. It also exposed a first-turn 5.31 s stable-boundary recomputation (SSD write itself 35 ms). This needs exact checkpoint reuse without changing forward segmentation; it is not an optimized tool-finalization result. The ordinary app AR cache receipts above do not cover that lane.

Cancellation during model loading, concurrent request admission, full-quota eviction, video/audio and other Nemotron bundles are not established by this checkpoint. The image-only bundle is not advertised as video/audio capable. Long-context kernel tuning remains a separate measured optimization task; no shared cache/sampling defaults were changed to improve these rows.

## Build notes
Use Swift's swiftbuild backend consistently. Engine tests use Release with DEBUG test helpers; app unit tests use the repository's Debug OsaurusCoreTests scheme. Release app builds are for live performance. A build failure before test execution is not a test failure or a pass.

Original Claude worktrees and Python reference source remain untouched. Temporary layer tracing is removed from deliverable source. Full diagnostic receipts, failed controls and source patches are retained privately; public PR comments summarize their provenance and limits.
