# Gemma 4 unified vision LayerNorm correctness — 2026-11-04

Status: scoped numerical correction qualified in a fresh native Osaurus build; normal PR review/merge remains pending. OCR and low-RAM family gates still fail. No release tags or publishing.

The three LayerNorms in the unified vision embedder incorrectly used the vision configuration's RMSNorm epsilon (1e-6). The reference uses LayerNorm's 1e-5 default independently of the RMSNorms. Only patch_ln1, patch_ln2 and pos_norm change. Ordinary Gemma vision towers, language RMSNorms, audio, sampling and generation defaults are unchanged.

Reference: [Gemma4UnifiedVisionEmbedder](https://github.com/huggingface/transformers/blob/3eb8e27a4f50da05e362ad19c0f3ab4a4eb17ead/src/transformers/models/gemma4_unified/modular_gemma4_unified.py#L854). The processor merge permutation was also reviewed; no patch-order defect was established.

The unified VLM cache identity gains `unified-vision-ln-1e-5-v1` so image-conditioned KV generated with the old math cannot be reused. This conservatively invalidates existing unified-VLM entries, including text entries under that model identity, once after update; other Gemma identities remain unchanged. No cache payload is deleted.

## Regression and source proof

Baseline runtime983d88617b59ac581cbbf9beb0f36de1a3bec786 fails both new tests: three production epsilon assertions, three numerical comparisons against independent CPU LayerNorm math, and the required changed cache identity. Low-variance numeric errors were0.20317/0.22369/0.22369. The regression inspects the actual production model's leaf LayerNorm modules, not a copied implementation. Tiny synthetic modules only; no source/BF16 model loaded. Corrected selection passed7XCTest tests with zero failures plus28SwiftTesting tests in3suites (parameterized cases counted separately). The first expanded attempt failed on missing rope_freqs_float32 because the top-level custom metallib was supplied; preserved failure was corrected by selecting the same app's full Cmlx bundled library through the existing test hook. Baseline was reproduced again with that full library, then the exact expanded selection passed. No gate or failing assertion was weakened.

## Native evidence and remaining acceptance

The existing Release Osaurus693034bab (full tree equals mergedb49dac53) with runtime983 loaded the exact12B QAT JANG_4M bundle, pagedRAMOff, prefix/SSDOn and explicitThinkingOff. Real image tensors and differing media fingerprints/salts were observed.

| Image | Expected approval / cents / banner | Observed |
|---|---|---|
| A | B739-AX42 /1847 /Blue | B730-AX4 /1847 /Blue |
| B | D205-KP96 /9203 /Red | B265-K96 /2823 /Red |

The same-media follow-up restored5814tokens but repeated wrong2823. Cache reuse is not a quality pass. These are retained failures; this epsilon correction is not yet claimed as their cause or complete fix. The fresh rebuilt-app results below complete the bounded tool/cache regression check; the image-reading defects remain open.

Earlier native reasoningOn/Off editor choices survived save/fullrestart and dispatched explicit enableThinking true/false. A genuine reasoning card appeared only forOn. Two real terminal calls waited for process exit and grounded a second command in random stdout; later diagnostic repeat restoredSSD on four requests. One earlier adaptive-mode next-user cache miss remains unexplained and is not retroactively fixed by a repeat hit.

Physical-footprint gates failed on the owned-loading runs. Retain owned loading per user choice and report RAM honestly; no low-RAM or family-ready claim. Other QAT JANG_4M/MXFP4 bundles, relevant cancellation/eviction/companion/scored evaluation rows remain partial. No audio, BF16/source, raw-speed benchmarking, default masking or forced behavior.

Retained evidence: `/Users/eric/vmlx-private-evidence/osaurus-cache-wiring-20261103/` contains `GEMMA-UNIFIED-NORM-BASELINE-RED-RECEIPT.json`, exact pinned upstream files and identity, `GEMMA-QAT-MEDIA-BASELINE-FAILURE-RECEIPT.json`, `GEMMA-QAT-TERMINAL-REASONING-NATIVE-RECEIPT.json`, and `GEMMA-QAT-CACHE-TRACE-REPEAT-RECEIPT.json`.

## Matched-profile native regression on the candidate

Source runtime `54c6c5b9faeee7d1b9739114a5b4eeba84672396` was built into Osaurus consumer `7cc4adbeeeaccd93c527565891a0428c2914e5d4`, version 0.25.32-dev/build 2026110412. The exact fresh Release executable SHA-256 is `71871872ec690b3f5233606956a6a17e11f718ff123346fbb900106068f75595`. The consumer's 115 focused tests across two suites and fresh Release build passed. This documentation follow-up changes no executable source from the qualified runtime head.

The independent agent/profile was frozen before inference and copied byte-for-byte into baseline and candidate fixtures. Both runs used the same two real, human-legible ICC-profiled PNGs, explicit Thinking Off, otherwise inherited bundle samplers, and identical prepared media tensors/fingerprints. The expected values were absent from the prompts. Results:

| Image | Matched baseline | Candidate |
|---|---|---|
| A, expected B739-AX42 / 1847 / Blue | B739-AX4 / 1873 / Blue | B739-AX4 / 1847 / Blue |
| B, expected D205-KP96 / 9203 / Red | B265-KP96 / 2232 / Red | B265-K96 / 2323 / Red |

Baseline scored 2/6 fields; candidate 3/6. **Both fail OCR.** These unseeded temperature-1 cases do not establish a causal quality improvement from this correction. Earlier results above use another profile and remain separate evidence. One accidentally restored Orchestrator tab was stopped and excluded; the contaminated fixture and failure receipt were retained, followed by a clean matched-profile baseline.

Three genuine calculator turns returned 437, then doubled that tool result to 874, then after a full process restart multiplied the previous result to 2622. All three completed with visible answers, parseable tool arguments/results and natural completion. The candidate's eight actual inference requests comprise two image answers and two requests per calculator turn. UI decode readouts were 40.9/40.5 tok/s for images and 40.9/40.4/40.7 for tool turns; runtime decode rates were 38.52–40.08 tok/s. JSONL end-to-end rates are separately retained, not conflated with decode. Baseline build contention and unfiltered clocks prevent a performance A/B claim.

Prefix and SSD cache were On; paged RAM was Off. The candidate used the new cache identity. All six tool-related requests restored disk prefixes; fresh-process restoration recovered 2840/2948 prompt tokens, followed by 2948/3004. Typed payload inspection at boundaries 2840, 2948, 3004 and 3031 found 48 layers: 8 raw KV and 40 rotating states, every rotating offset equal to its checkpoint boundary. Runtime reported disk-backed restore and effectiveKVLayers=0. Generic serialized kind 2 is not a TurboQuant layer claim. Native idle telemetry reported no active/queued work after model idle eviction. The last-sampler panel says compile requested despite global Compiled Decode being Off; this work makes no compiled-kernel claim.

Caller generation objects contained only explicit enableThinking=false. Bundle temperature 1/top-p .95/top-k64/min-p0 remained inherited; absent max output resolves to the existing application fallback 16384. No sampler, prompt or protocol-marker masking was added. All three app processes and their footprint collectors exited normally. Sampled physical-footprint peaks were 12,487,729,696 bytes baseline, 12,455,043,616 candidate and 11,592,360,984 after candidate restart, against 10,135,442,948 weight bytes: **all fail the low-RAM gate**. Owned loading is retained under the user's explicit decision.

SOURCE EVIDENCE: `Libraries/MLXVLM/Models/Gemma4.swift`, `Tests/MLXLMTests/Gemma4UnifiedVisionLayerNormTests.swift`, pinned upstream reference and the red/green test receipts.

LIVE EVIDENCE: `GEMMA-UNIFIED-NORM-NATIVE-QUALIFICATION-RECEIPT.json` in the retained evidence directory binds 56 artifacts including exact app/model identities, matched profiles, AX/screenshots, native JSONL export, runtime logs, typed SSD headers and finalized physical-footprint samples. No image fixture or user prompt was uploaded to GitHub.

Remaining family qualification: unresolved OCR, low-RAM failure, other QAT JANG_4M/MXFP4 bundles, relevant cancellation/eviction and scored evaluation. The earlier adaptive next-user cache miss is still unexplained. Mac/CUDA CI jobs were queued at the latest checkpoint; Linux/reference checks passed and repository-wide lint failed under its pre-existing advisory policy. No CI gate was weakened.
