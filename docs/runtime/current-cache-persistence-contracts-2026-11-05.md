# Current cache persistence contracts

The full focused target retained four issues across three growing-chat source
contracts. Investigation found stale assertions rather than a new runtime
fault: a third batch warmup gate now protects durable append-only SWA seed
reuse; disk seeds materialize append-only state before falling back to typed
snapshots; completed-answer storage permits canonical hybrid templates; and
DFlash's boundary call moved under the persistence/non-auxiliary guard.

Update those assertions to the effective policies while retaining the other
warmup, media, typed snapshot, stop/offset, and stable-boundary checks. Match
all DFlash helper arguments without treating indentation as behavior. Keep the
stable-boundary loop separate from the generated-answer guard. No production
cache, sampler, model, AgentLoop or app-tool implementation changes.

A new actual MLX checkpoint control captures nonzero full-KV plus rotating
state at nonaligned5/7tokens, grows both live caches by3tokens across another
ring wrap, then verifies exact prefix values, sealed rotating state and restored
offsets. Missing-layer/reset-state/old-boundary capture cases fail closed. Both
parameter cases pass on the unchanged runtime. The complete growing-chat
cache suite now passes30functions with no failures/skips. A final formatting
pass changes only whitespace; quoted literals and non-whitespace bytes match
the passing source exactly, with no new changed-line style diagnostics.

The full same-target candidate refresh records1250total/1204passed/9failed/
37skipped (1004SwiftTesting functions/13issues). No new failing method IDs
against the sampler baseline; four benchmark/cache source-contract failures
are cleared. The remaining nine independent failures stay registered and red. These are component controls, not model-family, batched
inference, speed, physical-RAM or multimodal qualification. Existing actual
app39 evidence remains valid for unchanged production; no new app behavior is
claimed from source guards. Canonical models remain read-only. No release/tag.

SOURCE EVIDENCE:
`Tests/MLXLMCommonFocusedTests/BatchEngineGrowingChatCacheSourceTests.swift`.
LIVE EVIDENCE: local `goal-storage-cache-resume-20261104/` raw baseline failures,
`cache-source-contract-triage/SOURCE-TRIAGE.json`,
`full-focused-xcode/append-only-checkpoint-baseline/Focused.xcresult`, and
`full-focused-xcode/cache-persistence-contract-candidate/Focused.xcresult`
plus frozen source/command/style bindings. No runtime defect is inferred from
the old source-string failures.
