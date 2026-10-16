# Nemotron 3.5 Super-VL integration checkpoint

This work is not merge-ready. Runtime and app validation remain in progress.

## Changes
- Preserve mixed affine projector quantization and validate the physical final vision LayerNorm tensors independently of speculative-decoding settings.
- Support image-only Omni bundles without constructing Parakeet audio modules.
- Match the bundle's image template expansion, rectangular pixel shuffle and float antialiased RGB preprocessing.
- Compute router logits and weighted expert sums in float32; retain float32 recurrent state and bound the SSM scan to 256-row chunks.
- Apply causal masking to cacheless language forwards.
- Use post-normalization states in the native MTP head and stage recurrent verify state for accepted-prefix commits.
- Share architecture eligibility between native-MTP activation and host controls. No new automatic tuning or sampler defaults are introduced.

## Evidence so far
- Coordinate-coded rectangular/square image shuffle: 2 tests pass.
- Image resizing against independently generated PyTorch vectors: 2 tests pass. Float32 CPE interpolation: 1 test passes.
- Physical final-norm tensor checks: 4 tests pass, including omitted/stale indexes and incomplete files.
- Exact-bundle image generation independently matches all 23 reference continuation tokens and ends at EOS. Vision first-eight L2 error is 0.26% with default TF32, 0.03% with TF32 disabled. This is a numeric diagnostic, not app proof.
- With TF32 disabled, all three text teacher-forced cases match at 100% of positions, with top-20 KL below 1e-12. Default TF32 produces 96.68–98.70% agreement; production precision policy has not changed.
- Staged Mamba rollback, gated normalization and fused convolution rounding tests pass (3/3). Native MTP depth-policy tests pass (7/7). The convolution retains its fused path and now preserves activation-dtype rounding at convolution, bias and SiLU boundaries.
- Isolated Release Osaurus: connected calculation/tool turns, motorcycle image recognition and image-history follow-up passed. Disk restore after restart and between tool steps includes 8 KV + 40 Mamba layers. Approximately 212 MB tool snapshots write in 21–22 ms in the background; the reply save path takes about 7 ms. Observed decode is 33.8–35.8 tok/s, with approximately 7.1 GiB peak physical footprint. These are functional rows, not controlled speed comparisons.

## Remaining acceptance work
- Resolve the independent greedy text difference (one case matches only five initial tokens) and image teacher-forced logit difference. A real convolution rounding defect was fixed, but it does not fully explain those differences. The Python vision fixture also used LayerNorm epsilon 1e-5 instead of the source bundle’s 1e-6; correcting that reference narrows embedding differences but does not close the teacher-forced gate.
- Native MTP On (Adaptive) currently resolves to AR for this bundle because usable tuning is absent; do not interpret the live AR rows as MTP proof.
- Complete cancellation, delegation/spawn and native-MTP connected app turns, with source-matched telemetry.
- Measure native MTP correctness, acceptance and actual throughput before assigning a default depth.
- App pin, app CI and public proof receipts before merge. No release/tag.
