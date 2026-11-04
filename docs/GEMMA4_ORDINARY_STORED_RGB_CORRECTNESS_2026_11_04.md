# Gemma 4 ordinary processor RGB and transparency — 2026-11-04

The ordinary Gemma 4 processor applied color management and preserved transparency instead of matching the upstream stored-RGB and white-alpha conversion contract. This correction shares the already qualified unified processor conversion with the ordinary processor. It preserves image geometry, model weights, sampling, templates, audio and decode kernels.

The pixel correction is qualified for the component contract and the named native E4B workflows below. CI and normal runtime/consumer landing remain separate steps. Other model bundles and broader family readiness remain partial. No release tags or publishing.

## Reference and production regression

The pinned primary [Gemma4ImageProcessor](https://github.com/huggingface/transformers/blob/3eb8e27a4f50da05e362ad19c0f3ab4a4eb17ead/src/transformers/models/gemma4/image_processing_gemma4.py) inherits the TorchvisionBackend RGB conversion contract: convert to RGB, rescale stored samples and do not normalize. The pinned backend delegates PIL RGB conversion to the white-alpha composite path without an ICC transform. Reference file identities are retained in `GEMMA-ORDINARY-PIXEL-UPSTREAM-IDENTITY.json`.

The actual `Gemma4Processor.prepare()` regression uses encoded PNGs in sRGB, Display P3 and Rec.2020, each opaque, half-transparent and transparent. Dimensions avoid resizing to isolate the pixel contract. Independent CPU stored-RGB/white-alpha math supplies expected values; repeated prepared media salts must remain stable. Both actual processor configuration classes are exercised without model weights.

On base `fe797ed49a0dbd28a0bf239af04430be35baf83f`, all nine unified cases pass. Ordinary sRGB opaque passes; its other eight cases fail. Maximum errors include 0.49996158 for half-alpha sRGB, 1.0 for fully transparent pixels, 0.6705315 for opaque P3, and 1.0248828 for opaque Rec.2020. The compatibility runner prints misleading testcase-passed lines alongside recorded issues; process exit 1 and final FAILED with 16 issue events are authoritative.

The correction passes 14 XCTest and 22 SwiftTesting tests across five selected suites. Ordinary 9/9 and unified 9/9 production pixel cases pass at the unchanged 2e-3 tolerance. The selection also checks media determinism/unmanaged rendering, media-salt restore, VLM cache scope, unified LayerNorm and paged-RAM default-off policy. These are component checks, not full resize or model-quality proof.

## Native E4B comparison and continuation

Native-qualified runtime implementation: `200bb81ac1fcc655013dda55a148856161ea78db`. Osaurus consumer source: `54b1939403672559287f4af210783d25ea654fc6`, with that revision in all four pins and two test expectations. Fresh Release build `2026110416` has executable SHA256 `b17d79b13781c98c715492f50fb3bc1f3c6fc523ac9cd69c473c01d78d80b034`; strict ad-hoc signature verification passes. The actual-pin consumer passes 115 focused tests across two suites.

Real model: `OsaurusAI/gemma-4-E4B-it-qat-JANG_4M`, with 10,789,234,410 weight bytes. Generation inherits bundle temperature 1 / top-p .95 / top-k 64, runtime min-p 0, no penalty/seed/max override, and explicit Thinking Off. Native effective sampler confirms top-k 64 and the existing app fallback maximum 16,384.

Two isolated profiles initially contain 123 byte-identical regular files totaling 2,641,132,992 bytes. The baseline uses the prior actual-pin app (`dc127b4d19e01e5623a04c2bfcd2a910a5709825` / runtime `fe797ed49a0dbd28a0bf239af04430be35baf83f`). Both native requests have identical sanitized input, tools and generation dictionaries and attach the same real transparent PNG. Insights deliberately redacts media bytes; this is request-structure equality, not a claim of exact wire-image byte identity.

The baseline reports approval and amount unavailable, with Green correct: 1/3 fields. The corrected app reads `G918-NZ37`, `6409` and Green: 3/3. Expected identifiers were absent from the prompt and previous answers. Two unchanged opaque slips also score 6/6, for 9/9 named image fields on the corrected path. This establishes the bounded transparency correction; it is not a broad OCR accuracy score.

Nine actual inference requests across six manual cases complete naturally. Actual calculator continuation produces `1767`, then `3534`, then `10602` after a complete app restart. Every request records a positive rate. Runtime decode rates are 61.49–68.80 tok/s; prefill-inclusive export rates are a different statistic. UI image rates are 70.7 / 69.0 / 68.8 tok/s and calculator rates 65.3 / 66.8 / 66.8 tok/s. These are correctness-run observations, not a controlled speed benchmark.

Changed media scopes cold-prefill. Subsequent unchanged six-image history has stable prepared pixels/media salt and actual disk reuse. The fresh process restores 5,568/5,797 prompt tokens, then 5,797/5,856 on the tool-result leg. Six actually restored checkpoint files are inspected: four raw KV plus twenty rotating layers, all rotating offsets equal their checkpoint, state dtype F16. Native UI visibly shows Prefix On, paged GPU cache Off and SSD On. This is disk-backed rotating/raw KV reuse, not TurboQuant-KV topology or fused-kernel proof. No old-eligible-cache upgrade negative control is claimed for the ordinary variant.

## Failures and remaining qualification

Physical footprint remains a failure: candidate peaks 12,307,539,472 and 11,965,965,648 bytes both exceed the 10,789,234,410 weight bytes. Owned packed loading is retained by the user's explicit decision; no low-RAM claim follows.

Earlier E4B advanced settings checks prove a saved top-k 32 and maxOutputTokens 1,000,000 survive relaunch and reach both real request legs. Resetting inheritance removes both fields from the next requests. This is wiring proof, not million-token generation capacity. A separate Thinking-Off turn unexpectedly contains a visible thought despite explicit false in the request and `thinking_mode=toggle_off` in runtime telemetry. Thinking On also produces reasoning. Full rendered-template/channel attribution remains open; this pixel change neither fixes nor hides it.

Other 26B/31B/MXFP4 bundles, a scored AgentLoop/Frontier harness, cancellation/eviction/quota, full resize parity, and wider family readiness remain partial. Unified 12B's earlier OCR 4/9 and RAM failures remain retained. No prompt coercion, forced reasoning boundaries or hidden sampler rescue was added.

SOURCE EVIDENCE: `Libraries/MLXVLM/Models/Gemma4.swift`; `Tests/MLXLMTests/Gemma4PixelContractTests.swift`; pinned upstream conversion references.

LIVE EVIDENCE: under `/Users/eric/vmlx-private-evidence/osaurus-cache-wiring-20261103/`, `GEMMA-ORDINARY-PIXEL-CONTRACT-BASELINE-RECEIPT.json` and `GEMMA-ORDINARY-PIXEL-CONTRACT-CORRECTED-RECEIPT.json` bind component red/green results. `GEMMA-ORDINARY-CANDIDATE-NATIVE-QUALIFICATION-RECEIPT.json` freezes 41 artifacts, nine actual requests and six native cases. `GEMMA-ORDINARY-MATCHED-BASELINE-NATIVE-RECEIPT.json` retains the failing control. All relevant app and memory-collector processes exited normally before receipt freezing. `GEMMA-E4B-QAT-PERSISTENCE-RESET-REASONING-NATIVE-RECEIPT.json` and `GEMMA-E4B-OFF-REASONING-RUNTIME-TRACE-CLARIFICATION.json` retain the separate advanced-setting observation.
