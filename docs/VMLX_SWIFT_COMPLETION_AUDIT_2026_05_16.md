# vMLX Swift Completion Audit - 2026-05-16

This audit turns `goal.md` into a concrete completion checklist for the
consolidated `vmlx-swift` engine. It is intentionally conservative: a row is
only `live-proven`, `unit-tested`, or `static-proven` when there is a real local
artifact or source reference. Everything else stays `open`.

Current pushed branch state:

- Branch: `vmlx-0.31.3`
- Pushed HEAD: `0fdb164` (`feat(runtime): add exact native mtp sampling`)
- Previous runtime commit: `9b1b254` (`feat(runtime): add tensor-gated qwen native mtp`)
- Current worktree is not clean because another agent is working on Flux-native
  Swift files. The dirty Flux files are excluded from this audit's commit scope.
- Current non-MTP focus: ZAYA1-VL template/tool/vision correctness,
  live multi-turn text rows, and VL media cache rows. Native MTP is parked for
  this pass.

## Objective Restated as Deliverables

The package is complete only when all of these are true:

1. `vmlx-swift` is the single Swift package surface Osaurus can pin instead of
   separately pinning `vmlx-swift-lm`, `mlx-swift`, `swift-transformers`, and
   Jinja.
2. All runtime surfaces have live or unit-test evidence: model loading, chat
   templates, generation config, reasoning modes, tool parsers, VL/audio/media,
   cache topology, continuous batching, TurboQuant/JANGTQ, MTP, and distributed
   stubs.
3. Every local model family under `~/models` has a status row with load,
   template, generation, multi-turn, cache, batching, speed, and result.
4. Cache behavior is topology-specific: no generic paged/prefix hit is accepted
   for DSV4 CSA/HSA/SWA, ZAYA CCA, Qwen hybrid SSM, VL/media, audio, or other
   path-dependent state.
5. No fake runtime guard is used to hide bad behavior. No hidden repetition
   penalty, no forced think close, no forced stop, no hidden reasoning downgrade,
   and no all-or-nothing MTP acceptance.
6. Osaurus wiring is mapped and later repinned only after the package evidence
   is strong enough.
7. All claims are pushed, committed, and documented with exact local artifact
   paths for raw logs that cannot be committed.

## Prompt-to-Artifact Checklist

| Requirement from `goal.md` | Current evidence | Status |
| --- | --- | --- |
| Check repo/process state before long tests. | `pgrep` checks before/after recent RunBench/test work; no `RunBench`/Swift build/test processes running after latest push. | live-proven |
| Do not overwrite unrelated work. | Current dirty files are Flux-owned (`Package.swift`, `Libraries/VMLX/VMLX.swift`, `Libraries/vMLXFlux*`, `Tests/vMLXFluxTests`, `tools/vMLXFluxProbe`) and were left unstaged. | live-proven |
| Package graph should consolidate MLX/Jinja/Transformers-style code. | `Package.swift` at pushed HEAD exposes products `MLX`, `Jinja`, `Tokenizers`, `MLXLMCommon`, `MLXLLM`, `MLXVLM`, `MLXPress`, and `VMLX`; `Libraries/VMLX/VMLX.swift` re-exports the stable Osaurus import surface. | static-proven |
| Osaurus current pin/wiring map. | `docs/VMLX_OSAURUS_CONSOLIDATION_AUDIT_2026_05_15.md`; current OsaurusCore pins are `mlx-swift=0a56f904`, `vmlx-swift-lm=2cc64dd`, `Jinja=58d21aa`, `swift-transformers=087a66b`. | static-proven |
| Compare `../vmlx-swift-lm`. | `docs/VMLX_SWIFT_LM_DEEP_COMPARISON_2026_05_15.md` plus updated consolidation audit. Sibling is dirty and behind origin; do not copy blindly. | static-proven |
| Local model inventory. | `docs/VMLX_LIVE_MODEL_MATRIX_2026_05_15.md`; local `find ~/models -maxdepth 4 -name config.json` shows DSV4, Hy3, Kimi, Laguna, MiniMax, ZAYA/ZAYA-VL, Qwen3.5/3.6, Gemma4, Ling, and Nemotron Omni families. | static-proven |
| Live infer matrix for installed models. | `docs/local/live-model-matrix/20260515Tinfer-under20/REPORT.md` and `status.tsv`. Many under-20GB rows passed; several rows failed or were skipped. | partial |
| MTP activation must use real tensor evidence, not names. | `9b1b254`; `docs/VMLX_SWIFT_MTP_OSAURUS_WIRING_2026_05_15.md`; local fail-closed artifact `docs/local/native-mtp-qwen36-20260515/jang4m-crack-native-mtp-denied-postrevert.log`. | live-proven |
| MTP depth-3 production acceleration. | Current D3 rows prove recursive draft plus accepted-prefix cache commit, but remain below the 50 tok/s production target. Small-M verifier tuning is still open. | open |
| Qwen3.6 MTP coherency and token/s. | D1 artifacts `jang4m-mtp-artifact-native-d1-postrevert-96.log` and `mx-mtp-artifact-native-d1-postrevert-96.log`: coherent, `30.2` and `34.0 tok/s` short rows. D3 prefix-commit artifacts under `docs/local/native-mtp-qwen36-20260516-d3-prefix-commit/`: coherent, `prefixCommit>0`, `rollbackRepair=0`, no loops. | live-proven for explicit MTP diagnostic rows |
| Qwen3.6 50 tok/s target. | Not achieved. Current rows are below target and the rejected verifier argmax experiment was not committed. | open |
| DSV4 native encoder, CSA/HSA/SWA, long context, vector drift. | Focused docs exist in `docs/VMLX_SWIFT_PRODUCTION_ENGINE_GATE.md`, but current full long-context/vector gate is still listed open in consolidation audit. | open |
| Prefix cache OFF/ON and cache hit proof. | Existing matrix/harness describes rows; not complete for every topology and model family. | open |
| Paged cache OFF/ON. | Existing focused tests and some model rows exist, but no package-wide matrix artifact proves all relevant architectures. | open |
| Disk L2 OFF/ON and fresh-session restore. | Existing docs and some rows exist; package-wide, per-topology proof remains incomplete. | open |
| SSM companion cache and async rederive. | Qwen/hybrid rows are required by docs; not exhaustively live-proven for all relevant local models. | open |
| VL media salt, same-image hit, changed-image miss. | `docs/local/live-model-matrix/20260516Tzaya-vl-think-template-fix/ZAYA1-VL-8B-JANGTQ4_vl_chat_cache.out`: same-media replay HIT, different-media MISS, coherent blue/orange follow-up. | live-proven for ZAYA1-VL JANGTQ4 |
| Nemotron Omni audio/Parakeet/RADIO. | `docs/local/live-model-matrix/20260515Tinfer-under20/` has Omni JANGTQ and JANGTQ4 `infer_omni` pass rows. Chunk-concat safety remains documented as not proven. | partial |
| Reasoning on/off/effort matrix. | Focused DSV4 pass-through exists; full model-family reasoning matrix is not complete. | open |
| Tool parser matrix by family. | DSV4 and selected templates have focused proof; full dsml/deepseek/gemma4/kimi/jang/zaya/llama/qwen/mistral matrix remains open. | open |
| Generation config defaults apply. | Harness supports resolved defaults; package-wide three-bundle proof and per-model override matrix remain incomplete. | open |
| Single-batch and continuous batching. | Batch harness exists; full B=1/B>1 overlap and isolation proof per family remains incomplete. | open |
| TurboQuant/JANGTQ encode/decode and acceleration toggles. | Focused JANGTQ/Hadamard/matmul proof exists; live low-footprint active routed expert pass for all relevant models remains open. | open |
| Distributed mode. | Targets exist (`MLXDistributed*`, `TPRankWorker`), but no no-peer distributed clean artifact is recorded for this audit. | open |
| Full `swift test`. | Current filtered test attempt still fails before focused tests because SwiftPM compiles `MLXPressPolicyTests` first and that target errors with `no such module 'Testing'`. This is a blocker, not a pass. | open |
| Release build. | `swift build -c release --product RunBench --jobs 2` passed after the non-MTP ZAYA1-VL template/harness changes. | live-proven |
| Osaurus single-package repin. | Not done. Osaurus still pins split runtime stack. | open |

## Current Model Matrix Snapshot

Local raw matrix: `docs/local/live-model-matrix/20260515Tinfer-under20/`.

Notable pass rows:

- Laguna XS.2 JANGTQ config/template/prod defaults.
- ZAYA text JANGTQ4, JANGTQ_K, and MXFP4 config/template/prod defaults.
- ZAYA-VL config/template and VL batch chat rows.
- Qwen3.5 35B A3B config/template/prod defaults/VL batch chat.
- Gemma4 JANG_4M config/template/prod defaults.
- Nemotron Omni JANGTQ/JANGTQ4 config/template/prod defaults/Omni rows.
- Qwen3.6 27B JANG_4M CRACK and JANG_4M MTP config/template/prod defaults/VL batch chat.
- Qwen3.6 27B MXFP4 CRACK config/template/prod defaults/VL batch chat.
- Qwen3.6 35B A3B JANGTQ config/template/prod defaults/VL batch chat.

Known failing rows from that snapshot:

- `ZAYA1-VL-8B-JANGTQ4_.infer_prod_defaults_cache_off` -> `fail:133`
- `ZAYA1-VL-8B-JANGTQ_K_.infer_prod_defaults_cache_off` -> `fail:133`
- `ZAYA1-VL-8B-MXFP4_.infer_prod_defaults_cache_off` -> `fail:133`
- `Qwen3.6-27B-MXFP4-MTP_.infer_prod_defaults_cache_off` -> `fail:133`
- `Qwen3.6-27B-MXFP4-MTP_.infer_vl_batch_chat` -> `fail:133`

2026-05-16 non-MTP ZAYA1-VL follow-up:

- Root cause for JANGTQ4/MXFP4 text failures was not sampling and not a model
  quality issue: the loader shim manufactured a Qwen-style `<think>` assistant
  prefill even though ZAYA1-VL bundles stamp `think_in_template=false` and their
  shipped VL sidecar generation prompt is plain `<|im_start|>assistant\n`.
- Fix scope: ZAYA1-VL metadata shim still materializes vision placeholders and
  Zyphra XML tools, but no longer emits `<think>` or `enable_thinking` rails.
  The production harness now probes the actual rendered prompt before requiring
  `.reasoning` deltas; non-toggleable templates must still produce visible,
  coherent output with no marker leak.
- `ZAYA1-VL-8B-JANGTQ4` now passes `BENCH_TEMPLATE_SMOKE=1`,
  `BENCH_PROD=1`, and `BENCH_VL_CHAT_CACHE=1`:
  `docs/local/live-model-matrix/20260516Tzaya-vl-think-template-fix/ZAYA1-VL-8B-JANGTQ4_template_smoke.out`,
  `.../ZAYA1-VL-8B-JANGTQ4_prod_postharness.out`, and
  `.../ZAYA1-VL-8B-JANGTQ4_vl_chat_cache.out`.
- `ZAYA1-VL-8B-MXFP4` now passes `BENCH_PROD=1` at
  `docs/local/live-model-matrix/20260516Tzaya-vl-think-template-fix/ZAYA1-VL-8B-MXFP4_prod.out`.
- `ZAYA1-VL-8B-JANGTQ_K` remains blocked: bundle-default sampling and explicit
  greedy both fail the same `7+8-11` row (`7`/`8` stochastic, `6` greedy) while
  other short rows pass. Artifacts:
  `docs/local/live-model-matrix/20260516Tzaya-vl-think-template-fix/ZAYA1-VL-8B-JANGTQ_K_prod.out`
  and `.../ZAYA1-VL-8B-JANGTQ_K_prod_greedy.out`. Do not call this variant
  production-ready until the quant/runtime cause is found.

Skipped rows because of the 20GB cutoff include DSV4 Flash, Hy3, Kimi,
MiniMax, Ling, and other large bundles. They are not production passes.

## Current MTP Boundary

Live-proven:

- MTP activation requires `VMLINUX_NATIVE_MTP=1`.
- Supported Qwen model types must expose real MTP tensor evidence.
- CRACK config metadata without tensors fails closed.
- Qwen3.6 MTP proof targets are
  `/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP` and
  `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP`.
- Qwen3.6 JANG_4M-MTP and MXFP4-MTP D1 smokes are coherent with telemetry.
- Qwen3.6 D3 prefix-commit smokes are coherent on both proof targets:
  `docs/local/native-mtp-qwen36-20260516-d3-prefix-commit/jang4m-mtp-d3-prefix-commit-no-checkpoint-256.log`
  and
  `docs/local/native-mtp-qwen36-20260516-d3-prefix-commit/mxfp4-mtp-d3-prefix-commit-no-checkpoint-256.log`.
  Both rows report `rollbackRepair=0`.
- MTP phase telemetry now reports target verify, MTP draft, sampling, and cache
  commit time. The 2026-05-16 phase rows show target verify dominates, so the
  next speed pass must focus on compiled/tuned small-M verifier execution.

Not yet complete:

- Production-speed depth-3 acceleration near the 50 tok/s target.
- Compiled/tuned small-M verifier.
- MTP with VL multi-turn/media salt/cache proof.
- Auto-launch eligibility for Osaurus.

## Immediate Next Gates

1. Continue diagnosing the remaining `fail:133` rows without MTP scope:
   `ZAYA1-VL-8B-JANGTQ_K` is still a real coherence failure on one math row.
   The JANGTQ4/MXFP4 ZAYA1-VL text-template failures are fixed and live-proven.
2. Run a clean-worktree focused test pass once Flux Package.swift edits are
   either committed by the Flux agent or isolated in a separate worktree.
3. Optimize the native MTP D3 verifier hot path. The correctness baseline now
   exists; the remaining production blocker is compiled/tuned small-M verifier
   execution near the 50 tok/s target.
4. Run DSV4 long-context/vector drift with current pushed engine.
5. Expand the model matrix beyond the 20GB cutoff for DSV4, MiniMax, Ling, Hy3,
   and Kimi one family at a time, with process checks before and after each run.

## Mergeability Call

Current state: **not release-ready and not complete**.

The package has useful pushed progress, especially consolidated product surface
work, MTP tensor-gated activation, and many local model smokes. It still has
uncovered or failing requirements in the model matrix, cache topology matrix,
full test graph, Osaurus repin, DSV4 long-context proof, VL text-path failures,
and production MTP D3 acceleration.
