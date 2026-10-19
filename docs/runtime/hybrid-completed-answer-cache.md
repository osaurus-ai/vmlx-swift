# Completed-answer SSD checkpoints for hybrid chats

Status: candidate; app proof pending.

A Release Osaurus Ornith 122B JANG_4M chat generated 10,384 tokens and stopped normally. Its next short recall request restored only 2,613 of 13,160 prompt tokens and spent 17.411 seconds in the prompt stage (17.681 seconds to first token). The previous turn logged `canonicalHybrid=true` and skipped completed-answer storage.

The stripped-prompt checkpoint remains essential for templates that rewrite assistant history. It does not justify suppressing a valid completed-answer checkpoint for templates that preserve it. The candidate retains both, while continuing to omit redundant exact-prompt records for canonical hybrid chats. Fetch chooses a checkpoint only by matching actual tokens and the existing semantic/media/model cache identity. No model-name exception is introduced.

Both solo and batch finalization use their existing completion, tool-call, offset, memory-budget, dtype and companion-state checks. No KV or recurrent state conversion changes; BF16 attention/conv and F32 GDN state remain as serialized. No changes to required-tool restore eligibility, quota policy, sampling, or adaptive governor. A template rewrite still falls back to the stripped checkpoint.

Costs and proof required: this adds one full valid completed-answer record per eligible answer. Measure its payload bytes, synchronous return, background write duration and RAM against reduced next-turn prefill; test quota pressure before claiming bounded full-quota behavior. The existing post-answer row classification learns promotion to resume-boundary protection from actual hits. Long-context and multimodal performance remain unqualified.

Source regression checks exercise actual token iteration and SSD serialization, compare restored recurrent state and dtype, reject an unforwarded answer and reuse the stripped fallback after changed history. A scheduler-path test additionally checks normal EOS and absence of an answer replay.

## Focused result

`swift test --build-system swiftbuild --filter HybridStripBoundaryPrefillTests`: 16 tests, zero failures. Solo checks compare F32 recurrent state after actual disk serialization and confirm changed-history fallback. The scheduler check completes through natural EOS and stores its completed-answer state without an additional answer replay; its existing stripped-prefix fallback replay remains. This is source/runtime-unit evidence, not Osaurus performance proof.

Native MTP already attempts post-answer storage with its own offset/trim guards. DFlash2 currently persists only a prompt or stripped checkpoint and ignores the generated-boundary flag. Neither speculative path is changed here; any extension needs its own target/drafter companion and rollback proof.
