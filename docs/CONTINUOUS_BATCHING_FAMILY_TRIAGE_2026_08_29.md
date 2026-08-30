# Continuous-batching family triage — 2026-08-29

## Status and purpose

This is the handoff for the PRs following vMLX PR #331. It separates four
failure surfaces that previously appeared as one “batching is broken” symptom:

1. a vMLX model/cache implementation that cannot represent B-wide state;
2. an Osaurus scheduler that requests more slots than the model can execute;
3. an OsaurusEvals fixture or scorer that assumes every model must execute B>1;
4. a model tool-template/parser failure that occurs before batching starts.

The overall family gate is **PARTIAL**. The Qwen3.5 crash is reversed on the
current PR head with a real AgentLoop row. DSV4 safe serialization works but the
eval assertion and RAM footprint are broken. Zaya never reached the batch
engine because all three AgentLoop trials emitted malformed tool arguments.
Flash-Next and the dense Gemma control still require live rows.

## Frozen proof provenance

- vMLX branch: `fix/qwen35-batch-position`
- vMLX proof head: `295f8c5c182b789d880a84e89b29c40ba2ac1b15`
- Qwen3.5 position fix: `8999e6d0f54ddd8b34079740e067093d9b3979dc`
- Osaurus proof base: `a1760574e`, with only the six vMLX pin sites repinned to
  `295f8c5c...`
- Release app build: `build/pr331-dd/Build/Products/Release/osaurus.app`
- Release app binary SHA-256:
  `30fd1787e4ce5863db41b2f3326d2e30eeb2fc502133bc0d41a2ac77741765fc`
- MLX metallib SHA-256:
  `24d4cfcd3ca8b15ead691e46219f35adabbea64c9f8de4eae9bf293fd8d5eb7b`
- Evidence root: `/private/tmp/vmlx-pr331-proof-295f8c5c`

GitHub Actions on the fork were skipped by the upstream-repository identity
guard. They are not CI evidence. Current source evidence is the clean vMLX
build (`swift build --build-tests`, RC 0), focused tests, and the live rows
below.

## Current family matrix

| Family / topology | Current source contract | Current live result | Classification | Next focused PR |
|---|---|---|---|---|
| Qwen3.5 / Ornith GatedDelta SSM + KV | Native B>1 through `BatchArraysCache` plus per-sequence KV offsets | **PASS on PR head**: exact former crash fixture completed 2/2 children in subwave `[2]`; 26.87 tok/s, 233 ms TTFT, 3346.67 MB peak footprint, process survived | vMLX crash fixed for this row; broader multi-turn/cache continuation still pending | Finish PR #331 proof with Release-app visual row and unequal-prefix/multi-turn continuation; do not broaden the position fix |
| Zaya CCA | Native B>1 through `BatchZayaCCACache` | **FAILED before batching**: 0/3 AgentLoop trials formed a valid `spawn_batch`; no execution wave, therefore no live CCA B=2 verdict | Tool-schema/template/parser/model-output lane first; batching result is **BLOCKED**, not failed | Add raw emitted-byte + resolved-template + resolved-parser telemetry; isolate parser transformation from model emission; then rerun AgentLoop and a raw two-request CCA row |
| DSV4 SWA + CSA/HSA pool | `maximumSupportedDecodeBatchSize = 1`; requests may queue but must serialize | Runtime safety **PASS**, eval row **FAIL**, RAM **FAIL**: both children completed, subwaves `[1,1]`, limited by `engineCapacity`, 13.51 tok/s, 185 ms TTFT; scorer expected `[2]`; peak footprint 104,993 MB | OsaurusEvals assertion bug and separate vMLX/runtime-memory defect | PR A: architecture-aware eval expectation and truthful requested-vs-effective telemetry. PR B: DSV4 allocation-retention/footprint investigation. Never make PR A waive the RAM gate |
| Qwen3.8 Flash-Next QSA + optional MTP | `maximumSupportedDecodeBatchSize = 1`; QSA itself preconditions B=1 | **UNVERIFIED on PR head** | Runtime cap is source-only until live proof | Run off/auto/manual-depth rows; require queued `[1,1]`, both children complete, request-local MTP telemetry, cache counters, tok/s, TTFT, footprint, process survival |
| Dense Gemma control | No architecture cap; native B>1 through `BatchKVCache` | **UNVERIFIED on PR head** | Needed to prove the new cap does not serialize capable dense models | Run exact two-worker fixture; require effective slots 2, subwave `[2]`, coherent final, bundle generation config, tok/s, TTFT, footprint and cache telemetry |

## Source trace by failure boundary

### 1. Qwen3.5 crash: vMLX runtime bug

The old Qwen3.5 VLM language path treated the `BatchKVCache` graph offset as a
scalar. At B=2 the offset is a two-element vector, so `reshaped([])` trapped:

```text
[reshape] Cannot reshape array of size 2 into shape ().
```

The crash report at
`~/Library/Logs/DiagnosticReports/osaurus-evals-2026-08-29-010159.ips`
traces `MLXArray.reshaped` to
`Qwen35Language.LanguageModel.resolvedPositionIds` and then
`BatchEngine.stepBatchDecode`. Commit `8999e6d0` preserves the offset vector as
`int32`. The regression prefills two unequal sequences, decodes them together,
requires logits shape `[2, 1, 100]`, and verifies independent offsets advance
from `[3, 2]` to `[4, 3]`.

This is a runtime/cache bug. OsaurusEvals merely exposed it.

### 2. DSV4 safe serialization: runtime correct, eval contract wrong

DSV4 now advertises `maximumSupportedDecodeBatchSize = 1`. `BatchEngine`
clamps both construction and later resize requests to that model limit. The
live AgentLoop row requested two concurrent sequences, but the execution
envelope truthfully reported:

```text
effective_local_slots = 1
local_subwaves = [1, 1]
limited_by = ["engineCapacity"]
succeeded = 2
failed = 0
```

Both children returned the required tokens and the parent exited
`finalResponse`. That is a safe-serialization pass, not a native B=2 pass.

The eval fixture currently hardcodes `effectiveLocalSlots: 2` and
`localSubwaves: [2]`. Its setup note also calls
`InferenceFeatureFlags.mlxBatchEngineMaxBatchSize` “effective,” even though it
is only the requested/configured value and does not include the model cap.
Therefore the eval fails behavior that is correct for this architecture.

The eval PR must:

- preserve a strict native-B>1 fixture for architectures that claim B>1;
- add an explicit safe-serialization expectation for capped architectures;
- record requested width, model maximum, engine-effective width, observed
  local slots, subwaves, and limiting factors as distinct fields;
- reject a serialized row if any two children overlap inside a B=1 model
  forward, either child fails, or the process exits;
- never convert DSV4's 104,993 MB footprint into a pass merely because the
  behavioral expectation is corrected.

Evidence:

- `dsv4-serialized.json`
- `dsv4-serialized.log`
- `dsv4-serialized-footprint.csv`
- `dsv4-serialized.transcripts/`

### 3. Zaya: the AgentLoop row did not test batching

Across three trials, the first `spawn_batch` arguments contained schema-like
XML placeholders instead of a JSON `jobs` array, for example:

```json
{"job_id":"writing","jobs":"<element>\n<name>job_id</name>..."}
```

Later calls repeated `<value>...</value>` placeholders, guessed unavailable
tools, or reached the iteration cap. Since no valid `spawn_batch` executed,
these runs say nothing about `BatchZayaCCACache` under B=2.

Do not immediately call this a cache bug or a parser bug. The next diagnostic
PR must capture all three boundaries for the same step:

1. exact rendered tool-schema/template bytes given to the model;
2. exact raw model bytes before tool parsing;
3. parsed tool name/arguments plus the resolved `ToolCallFormat` and its source
   (explicit override, JANG capability stamp, or heuristic).

Interpretation:

- malformed schema in (1) means Osaurus schema/template composition is broken;
- valid schema in (1), malformed raw output in (2) means the bundle template,
  model capability, or model behavior needs correction/evaluation;
- valid raw tool syntax in (2), malformed arguments in (3) means the vMLX tool
  parser is broken.

Only after that boundary is fixed should AgentLoop be used as the CCA B=2 gate.
In parallel, a raw engine-level two-request row should exercise CCA
gather/scatter without tools so parser quality cannot mask a cache failure.

Evidence:

- `zaya-cb2-repeat3.json`
- `zaya-cb2-repeat3.log`
- `zaya-cb2-repeat3.transcripts/`

### 4. Flash-Next QSA and MTP: safe width is not yet a live result

`Qwen4ExpQSAIndexer` currently preconditions `B == 1`. The model-level cap
prevents the scheduler from feeding it a wider batch, but this remains only
source and unit evidence until the real model passes:

- MTP off;
- MTP auto, with activation justified by measured tuning;
- an explicit supported depth (currently depth 2 in the proof plan).

For every row, record requested and effective widths, queue/subwave shape,
both child outcomes, MTP configured and active depth, verify/accepted/rejected
or AR counts, cache topology/counters, TTFT, token/s, peak physical footprint,
stop reason, and final visible coherency. MTP telemetry from one queued request
must never bleed into the next.

## Required PR sequence

### PR 1 — finish vMLX #331 narrowly

- Keep the Qwen3.5 vector-position correction.
- Keep the general model-declared maximum decode width and the DSV4/QSA caps.
- Retain the behavioral clamp test.
- Run the remaining Flash-Next and dense-Gemma rows.
- Obtain one exact Release-app visual proof with one app instance; CLI/evals
  remain paired diagnosis, not GUI proof.
- If Zaya's parser lane cannot be repaired without broadening #331, document it
  as a blocking follow-up rather than mixing an unrelated parser change into
  the Qwen3.5 runtime fix.

### PR 2 — OsaurusEvals architecture-aware batching contracts

- Split native-parallel and safe-serialized expectations.
- Replace the misleading static “effectiveMaxBatchSize” note with actual
  runtime capacity telemetry.
- Keep failure semantics strict for wrong width, missing child, overlap under a
  cap, crash, malformed execution envelope, missing tok/s, or footprint breach.
- Add fixture/scorer tests for a native `[2]` row and capped `[1,1]` row.

### PR 3 — Zaya tool boundary and CCA live gate

- Add the three-boundary trace described above.
- Fix only the boundary proven wrong.
- Re-run three AgentLoop trials plus raw B=2 unequal-prefix CCA decode,
  multi-turn continuation, and cache restore/isolation.

### PR 4 — DSV4 footprint retention

- Treat 104,993 MB as a failed row.
- Attribute retained memory by weights, prefill activations, SWA cache,
  CSA/HSA pool, compiled graphs, disk-L2 staging, and per-child lifetime.
- Sample `phys_footprint` throughout warm-up, parent prefill, child 1, child 2,
  parent continuation, cache store, and unload.
- Prove old per-request state is released before claiming the serialization
  path is production-safe on constrained hosts.

### PR 5 — Osaurus repin and app proof

After vMLX changes merge, repin every Osaurus vMLX lock site to the merged SHA,
build the Release app, verify one instance, and drive the actual UI. Re-run the
family matrix from the repinned app/runtime path. Do not use HTTP-only results
as a substitute for rendered app behavior.

## Acceptance checklist for every live family row

- exact app/eval binary and vMLX SHA recorded;
- exact model ID and bundle generation config used, with no hidden sampler or
  reasoning coercion;
- requested width, architecture limit, engine-effective width, observed slots,
  subwaves, and limiting factors;
- all child results and parent final response coherent;
- multi-turn continuation, stop/cancel, and restart/disk-restore where the
  family gate requires them;
- TTFT and decode token/s;
- `phys_footprint`, not only allocator/RSS estimates;
- prefix/paged/disk-L2 counters and architecture companion state (SSM, CCA,
  QSA/MTP, media) as applicable;
- no marker leakage, loop, silent terminal row, length-cap fake pass, crash, or
  stale cache reuse;
- real image/video payload and cache reuse for VL/video claims;
- Release-app visual confirmation before calling user-facing behavior fixed.

Anything missing is `PARTIAL` or `BLOCKED`, with the artifact and reason named.

## Ornith 1.5 35B decode-performance follow-up

The MTP-MoE load fix did not explain the reported roughly 25 tok/s decode
rate.  On merged vMLX `aee2a8e0`, the production Qwen3.5-VL decoder left its
existing compiled GDN and MoE decode regions disabled for this exact
architecture.  Enabling those regions only for the measured model topology,
while retaining an explicit environment opt-out, is the causal speed change.
The policy is based on the decoded architecture fields rather than a bundle
name, so neighboring Qwen3.5, Qwen3.8, Gemma, Laguna, and DSV4 configurations
do not inherit it.

Current unmerged proof from `fix/ornith35-fused-affine-moe`:

- Release, 128 generated tokens: median **94.0 tok/s**
  (`94.2/94.0/92.7`), TTFT 164-169 ms, peak physical footprint 27,393 MiB;
- Release, 512 generated tokens: median **91.5 tok/s**
  (`90.9/91.5/91.5`), TTFT 168-169 ms, peak physical footprint 27,315 MiB;
- output was coherent and byte-stable across each greedy three-run row, with
  no loop, unclosed reasoning, or protocol-marker leak;
- the three-turn tool row produced a structured `xmlFunction` call, consumed
  its result, summarized it, and recalled the launch ID on the next turn;
- hybrid cache telemetry recorded paged hit 1, SSM hits 2/re-derives 2, and
  seven disk stores;
- cross-session disk restore matched 138/138 prompt tokens and reduced prompt
  processing from 501 ms cold to 29 ms warm;
- focused numerical tests pass for both SiLU/sigmoid compiled GDN tails and
  the real q5/q5/q4 Ornith expert topology.

Artifacts:

- `/private/tmp/ornith35-final-release-default-128.log`
- `/private/tmp/ornith35-final-release-default-512.log`
- `/private/tmp/ornith35-final-agentic-tool.log`
- `/private/tmp/ornith35-final-disk-restore.log`
- `/private/tmp/ornith35-compiled-policy-tests2.log`
- `/private/tmp/ornith35-fused-moe-tests-final.log`

This lane remains `PARTIAL` until the focused branch is reviewed in its own PR
and current-head CI passes.  The rows above are engine/runtime proof, not a
Release Osaurus UI proof.
