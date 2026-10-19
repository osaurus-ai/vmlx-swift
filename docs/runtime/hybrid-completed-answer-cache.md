# Completed-answer SSD checkpoints for hybrid chats

Status: candidate; focused tests and connected Release-app checkpoint reuse pass. Long-context/quota and speculative-family coverage remain open.

A Release Osaurus Ornith 122B JANG_4M chat generated 10,384 tokens and stopped normally. Its next short recall request restored only 2,613 of 13,160 prompt tokens and spent 17.411 seconds in the prompt stage (17.681 seconds from generation submission to first token, excluding a preceding 5.4-second model reload). The previous turn logged `canonicalHybrid=true` and skipped completed-answer storage.

The stripped-prompt checkpoint remains essential for templates that rewrite assistant history. It does not justify suppressing a valid completed-answer checkpoint for templates that preserve it. The candidate retains both, while continuing to omit redundant exact-prompt records for canonical hybrid chats. Fetch chooses a checkpoint only by matching actual tokens and the existing semantic/media/model cache identity. No model-name exception is introduced.

Both solo and batch finalization use their existing completion, tool-call, offset, memory-budget, dtype and companion-state checks. No KV or recurrent state conversion changes; BF16 attention/conv and F32 GDN state remain as serialized. No changes to required-tool restore eligibility, quota policy, sampling, or adaptive governor. A template rewrite still falls back to the stripped checkpoint.

Costs and proof required: this adds one full valid completed-answer record per eligible answer. Measure its payload bytes, synchronous return, background write duration and RAM against reduced next-turn prefill; test quota pressure before claiming bounded full-quota behavior. The existing post-answer row classification learns promotion to resume-boundary protection from actual hits. Long-context and multimodal performance remain unqualified.

Source regression checks exercise actual token iteration and SSD serialization, compare restored recurrent state and dtype, reject an unforwarded answer and reuse the stripped fallback after changed history. A scheduler-path test additionally checks normal EOS and absence of an answer replay.

## Focused result

`swift test --build-system swiftbuild --filter HybridStripBoundaryPrefillTests`: 16 tests, zero failures. Solo checks compare F32 recurrent state after actual disk serialization and confirm changed-history fallback. The scheduler check completes through natural EOS and stores its completed-answer state without an additional answer replay; its existing stripped-prefix fallback replay remains. This is source/runtime-unit evidence, not Osaurus performance proof.

Native MTP already attempts post-answer storage with its own offset/trim guards. DFlash2 currently persists only a prompt or stripped checkpoint and ignores the generated-boundary flag. Neither speculative path is changed here; any extension needs its own target/drafter companion and rollback proof.

## Connected Release-app result

App source `372699a9d66dd164efd31a890f027b6fbdc55768`, engine `d62c002b5e1cac583abbf03bb66a0f4a29405e2c`, M5 Max 128 GB, Ornith 122B JANG_4M with the explicit raw-norm metadata correction. No concurrent compilation. Bundle sampling unchanged; these requests ran AR, despite the saved Adaptive setting, because the bundle lacks qualified adaptive tuning.

A naturally completed 553-token answer stored both its 13,361-token stripped fallback and a 13,916-token completed-answer checkpoint. The next 14,079-token prompt restored the latter from disk (`postAnswerRowsResume=true`), processing only 163 new tokens. The visible answer correctly recalled all three previous topics and the earlier review label.

- Follow-up prompt stage: 453 ms; generation-submit TTFT: 861 ms; 100 generated tokens, 31.7 tok/s reported generation rate (33.8 tok/s excluding first-token interval). This is not a matched before/after speed ratio and not a 128k result.
- Additional checkpoint: 495,649,544 payload bytes; synchronous disk-store return 16.2 ms, background write 296 ms, with 245 ms queue delay behind the fallback. The next turn's two writes took 306/330 ms. These are serialized writer timings, not raw hardware SSD bandwidth.
- Header inspection: all 36 recurrent tensors remain F32 `[1,64,128,128]`, totaling 150,994,944 bytes; attention/conv remain BF16. No FP16 conversion was introduced.
- Sampled process physical footprint peaked at 3.58/5.48 GB on the two turns, with zero wire rejections. This excludes file-backed weight residency and is not a total model-RAM claim; the bundle has 73.88 GB of weights.
- First turn's model reload took 14.71 s, separately from its 806 ms generation-submit TTFT. Do not present the latter as Send-to-first-output latency.

Quality limitation: the longer answer incorrectly claimed that removing odd items from `[1,2,2,3]` during iteration leaves `[2,2,3]`; executing that snippet leaves `[2,2]`. Cache recall passed, but the whole generated code explanation is not an accuracy pass. Runtime causation is not established.
