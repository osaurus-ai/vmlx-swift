# Ornith DFlash2 adaptive loading and checkpoint audit

Status: implementation in review; app qualification incomplete. No release or model upload.

## Reproduced failure

On engine main 5bbcaf8, selecting Ornith 1.5 122B JANG_4M with its compatible bundled DFlash2 and On (Adaptive) crashes during load warm-up. Two isolated Release app runs terminate with SIGTRAP in `vlmGatedDeltaTree` (`Qwen35.swift`, precondition W <= 16). The adaptive chooser selects a 32-row tree, while the kernel reserves 16 rows. Headers and successful discovery did not exercise this path.

The correction sizes the kernel register workspace for the selected tree. It leaves the chooser, sampling and drafter weights unchanged. Branching and chain equality tests now include 7, 16 and 32 rows.

## Additional ownership and loading defects

- DFlash2TokenIterator retained a full-prompt checkpoint even when the canonical hybrid boundary was the only checkpoint its terminal store consumed. Omit the unused copy. Transfer the captured hybrid checkpoint through the existing disk-only admission helper after prefill; release iterator ownership only after acceptance. A refused publication retains the checkpoint for terminal retry. Writer acceptance is not disk durability; pending writers still own arrays.
- Load configuration independently enabled native MTP even when external DFlash2 won strategy selection. Resolve both from the same bundle directory and selected strategy so an unused native head is scrubbed. An absent/incompatible drafter still resolves through the existing native policy; Off disables both.

## Qualification

- Kernel suite: 11 tests passed, including bit-exact tree/chain paths at 7/16/32 rows.
- First cache suite: 9 of 10 passed; an asynchronous test incorrectly expected a writer to remain pending after initialization. It had already settled. Corrected the assertion to verify the relevant contract: no iterator-owned checkpoints, durable exact recurrent/KV restore, and no duplicate terminal write. Final cache suite: 10/10 passed; shared early-spill suite: 4/4 passed. Load-selection assertions passed except a URL trailing-slash fixture comparison, corrected before rerun. Broader settings suite: 44/45 passed; unchanged cache-size test expects 42GB and fails under current quota policy. Not counted as passing.
- Rebuilt Release app at engine c73225e9f / app 3f3f61d9 completed the formerly crashing adaptive 32-row path. Two visible Helper turns ended normally: code 970 tokens at 49.10 tok/s, follow-up 242 at 45.20 tok/s. Physical peaks 6.27 and 6.90 GB. Traced diagnostic rows, not a matched speedup claim.
- First turn restored 2,642/3,815 prompt tokens; follow-up restored 3,814/6,087. Canonical checkpoint writes settled at 28/50 ms (247/303 MB); iterator retained no checkpoint at decode-end. The existing canonical-only policy reprocesses the prior answer on the next turn; post-answer reuse remains open.
- Load selection rerun: 25 passed, 1 intentional skip, 0 failures. Resolver ownership tests: 3/3; cache regression rerun: 10/10. Resolver changes and aggregate telemetry below are not yet in this app binary.
- Matched Auto/Off prose/code speed, long context and further app qualification remain pending.
- JANGH2, DSpark/MiniCPM, Nemotron, concurrent/spawned slots, media, cancellation and full-quota eviction remain separate qualification rows. Shared source does not prove all families.

Private evidence: `~/vmlx-private-evidence/ornith-dflash2-adaptive-20261020/LEDGER.md`, `baseline/` (two crash reports, process exit receipt, app logs, physical-footprint samples), focused test logs. No successful generation or speedup claim from the failed warm-up rows.

## Concurrent drafter ownership

Two simultaneous cache misses previously loaded separate full drafters. Loading now uses one in-flight record per canonical directory; unrelated directories can load concurrently. Eviction removes publication rights, so an older completion cannot repopulate the cache or replace a newer load. Existing request owners still receive and own their result. Focused tests cover same-path sharing, independent paths, eviction/evictAll during loading and failed-load retry.

The diagnostic decode-end memory row now includes aggregate adaptive widths, acceptance, AR tokens, draft/verify/commit/AR time on ordinary stop, without requiring per-cycle tracing.

## Full-model tree verification correction

The live crash correction was necessary but insufficient. An actual-weight diagnostic compared every tree row with a single-token target forward over that row's exact ancestor path. The former routed Qwen tree path differed at widths4/16/32 (chain argmax3/4,12/16,25/32; branching also differed). A coherent sampled app answer did not detect this.

Routed Qwen3.5 tree verification now uses its existing row-exact projection/expert routes and GDN tail. Wide trees partition independent projection/MLP operations into the existing qualified8row tiles; the draft tree and adaptive width remain unchanged. Attention uses AR's single-query computation over precisely the prefix and ancestors. Contiguous paths use views; branched-path KV temporaries settle one row at a time so they cannot accumulate across the tree. This has a copy/launch cost that still needs long-context app measurement. Dense Qwen tree routes, native-MTP width admission and ordinary prefill are unchanged.

Actual bundles: Ornith1.5-122B JANG_4M and JANGH2 each match104/104 logits rows bit-exactly at widths4/16/32, chain and branching. All six post-commit next-token comparisons match exactly for each bundle. Scope tests6/6 and kernel tests11/11 pass. `RunBench/RowCostBench.swift` exposes the diagnostic via `BENCH_ROWCOST_COMPARE_TREE=1`; this is numeric verification, never app speed proof.

Artifacts: `tree-target-parity.log` (before), `tree-target-exact-v1.log` (insufficient projection/tail correction), `tree-layer-parity.log` (first divergence at attention layer3), `tree-attention-parity.log`, `tree-target-exact-v2.log` (4M), `tree-target-h2-exact.log`, `tree-correction-tests.log`. The earlier app speed numbers predate this correctness correction and cannot qualify its speed.
