# Gemma 4 unified vision LayerNorm correctness — 2026-11-04

Status: candidate; numeric baseline reproduced and scoped corrected tests passed; native qualification and CI pending. No release tags or publishing.

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

The same-media follow-up restored5814tokens but repeated wrong2823. Cache reuse is not a quality pass. These are retained failures; this epsilon correction is not yet claimed as their cause or complete fix. Fresh rebuilt-app image, text/tool, growing-cache and process-restart proof remains required, with independent profile/source/model identity.

Earlier native reasoningOn/Off editor choices survived save/fullrestart and dispatched explicit enableThinking true/false. A genuine reasoning card appeared only forOn. Two real terminal calls waited for process exit and grounded a second command in random stdout; later diagnostic repeat restoredSSD on four requests. One earlier adaptive-mode next-user cache miss remains unexplained and is not retroactively fixed by a repeat hit.

Physical-footprint gates failed on the owned-loading runs. Retain owned loading per user choice and report RAM honestly; no low-RAM or family-ready claim. Other QAT JANG_4M/MXFP4 bundles, relevant cancellation/eviction/companion/scored evaluation rows remain partial. No audio, BF16/source, raw-speed benchmarking, default masking or forced behavior.

Retained evidence: `/Users/eric/vmlx-private-evidence/osaurus-cache-wiring-20261103/` contains `GEMMA-UNIFIED-NORM-BASELINE-RED-RECEIPT.json`, exact pinned upstream files and identity, `GEMMA-QAT-MEDIA-BASELINE-FAILURE-RECEIPT.json`, `GEMMA-QAT-TERMINAL-REASONING-NATIVE-RECEIPT.json`, and `GEMMA-QAT-CACHE-TRACE-REPEAT-RECEIPT.json`.
