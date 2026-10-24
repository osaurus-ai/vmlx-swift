# D3 decision runtime: scope and qualification

This change adds metadata-detected D3, D3 Flash and D3 Turbo decision backbones, including mixed affine 4/6/8-bit and dense mixed-bit Hadamard bundles. The live qualification record is [engine PR #645](https://github.com/osaurus-ai/vmlx-swift/pull/645): its source/artifact-keyed proof comments are authoritative. Private raw logs and the investigation ledger are in `~/vmlx-private-evidence/d3-swift-20261023/`. A source checklist, metadata check or diagnostic executable alone is not application proof.

## Implemented contract

- Strict decision metadata, readout SHA256/float32 shape and tokenizer answer-code admission. Declared decision bundles cannot enter chat factories, even when incomplete.
- Headless Qwen3.5 backbone: noncausal full attention for D3/Flash; prefix-bidirectional full attention for Turbo; causal GDN in both. Exact query-only attention chunks retain all keys and bound the head-256 fallback workspace.
- Declared Qwen2 tokenization, NFC byte boundaries, ordered JSON/options, typed choice/noul/score results, and Turbo image-box readouts/confidence.
- State-level images/videos with the vendor decision pixel/frame policy, uint8 bicubic processing, timestamp-separated video groups and corrected Qwen3.5 multimodal positions. D3 uses its declared GELU-tanh vision MLP.
- Existing per-module affine and mixed Hadamard preparation, normal norm convention resolution, scheduling limits, allocator policy and residency leases. No sampler, autoregressive iterator, speculative head or conversational KV/SSD checkpoint is created.

Merge qualification requires same-weight numeric fixtures by family/quant/media type, focused regressions, fresh Osaurus-owned API/UI proof, repeated-request/cancellation resource cleanup, and required CI. Prompt throughput and peak/settled physical footprint must be distinguished from generated token/s, MLX active bytes and allocator pool bytes. Peak memory includes real prompt/media activations; no weights-only peak claim is made.

## Findings retained for reviewers

The first sweep found three H2 admission failures and an oversized/resampled video prompt. The metadata exception is narrow: the authoritative JANGH v2 plan stays in config.json, while a headless D3 summary may omit the older version summary. Negative metadata-conflict tests remain mandatory.

Further runs caught an HWC/NCHW integration error, raw-vs-NFC state-boundary mismatch, sigmoid-vs-tanh GELU mismatch, and two large images using the checkpoint's generic 16.8M-pixel budget instead of the vendor decision API's 1.64M budget. These are real failures, with retained artifacts; initial smoke agreement never qualified them. The corrected processor configuration is shared with the diagnostic path and covered by focused regressions.

An independently executed reference A/B also found that installed Python casts fractional position-interpolation weights to the packed quantized embedding dtype. Swift keeps floating interpolation rather than reproducing that bug. Fixture comparisons must disclose this reference distinction, especially on media. `MLX_ENABLE_TF32=0` did not change the diagnostic text vectors and was rejected as an explanation; no process-wide precision override was added.

## Explicit follow-ups

- The first Osaurus endpoint supports choice/noul/score, state-level encoded images/videos and Turbo image boxes. Vendor asset-reference composition, video-at/track and StateOnce are separate extensions, not qualified features of that endpoint.
- D3 media/activation corrections are opt-in. Generic generative timestamped-video positioning and declared vision activation need their own regression lane; this PR makes no all-model repair claim.
- Existing bundle manifests say minimum Osaurus 0.25.26, but that installer predates this implementation. A minimum-version stamp alone is not capability proof; the first packaged capable build and publication metadata must be reconciled before announcing stable support.

No release tag, model upload or published-bundle modification is part of this change.
