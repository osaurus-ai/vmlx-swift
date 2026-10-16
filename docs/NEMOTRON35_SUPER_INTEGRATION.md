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
- Staged Mamba state test passes with TF32 disabled. Its initial randomized float32 projection compared multirow TF32 GEMM against serial float32 GEMV; deterministic exactly representable projection coverage is being added to isolate state correctness.

## Remaining acceptance work
- Resolve the independent greedy text difference (one case matches only five initial tokens) and the image teacher-forced logit difference. Disabling the fused convolution did not resolve the former; no convolution change was made.
- Execute corrected focused tests on the final source, including gated-normalization test setup and native-MTP control eligibility.
- Real Osaurus connected text/tool/image turns, disk restore with matching KV and recurrent state, cancellation and reload, physical footprint, prefill/decode/TTFT and SSD read/write measurements.
- Measure native MTP correctness, acceptance and actual throughput before assigning a default depth.
- App pin, app CI and public proof receipts before merge. No release/tag.
