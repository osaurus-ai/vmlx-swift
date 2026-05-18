# vMLX Swift MTP / Osaurus Wiring Plan - 2026-05-15

This document records the Swift-side MTP status and activation contract for
Osaurus. Native MTP now exists as an explicit, tensor-gated runtime path for
Qwen3.6, but it is still not an automatic production launch mode. Auto-launch
requires the cache, VL, multi-turn, and speed gates below.

## 2026-05-17 Matrix Update

The current six-variant Qwen3.6 Swift matrix is recorded in
`docs/VMLX_QWEN36_MTP_MATRIX_2026_05_17.md`.

Key changes since this plan was first written:

- All six local Qwen3.6 MTP/VL bundles are now present:
  27B JANG_4M, 27B MXFP4, 27B MXFP8, 35B JANG_2K, 35B MXFP4, and 35B MXFP8.
- MTP activation is tensor- and tuning-gated. Supported Qwen MXFP/JANG_4M MTP
  bundles now resolve auto native depth from real weight evidence plus the
  bundle-local `vmlx_mtp_tuning.json`; JANG_2K remains blocked by its tuning row.
- Text MTP speed rows now clear the target for the MXFP artifacts where the
  current gate selected MTP: 27B MXFP4 D3, 35B MXFP4 D3, and 35B MXFP8 D3.
  27B MXFP8 currently prefers D2 on the count prompt. 35B JANG_2K remains
  blocked and should stay AR-only.
- Exact-cache repeat and growing-chat rows hit disk L2 plus SSM companion state.
  Qwen hybrid/SSM native MTP auto-launch remains limited to the exclusive
  single-request path until raw B>1 scheduling is implemented.
- VL+MTP must use `BatchEngine.generate`/`Evaluate.generate` exclusive paths.
  `BatchEngine.submit` raw native-MTP scheduling is intentionally rejected until
  per-slot draft/verify/cache scheduling is implemented.
- Strict VL+MTP currently passes the MXFP production rows when given sufficient
  budget, including the 35B MXFP4 red/blue image row. 35B JANG_2K remains a
  blocked profile.
- Bundle-default stochastic exact-p/q over hybrid SSM now resolves to
  `verifierMode=sequential_repair`; forcing the fast chunk verifier through
  non-greedy hybrid rows reproduced a real 35B MXFP4 growing-chat failure.
  Greedy chunk commit remains an explicit diagnostic/speed path where proven.

## Current Boundary

The package now has a no-load MTP inspector:

- `MTPBundleInspector.inspect(modelDirectory:jangConfig:)`
- `MTPBundleStatus`
- `MTPRuntimeMode`
- `MTPDraftStateContract`

The inspector reads only metadata and tensor names:

- `config.json`
- `jang_config.json`
- `model.safetensors.index.json`
- safetensors headers when no index file exists

It never materializes tensors and it does not alter generation. Plain
autoregressive decode remains the default path unless an explicit native MTP
request passes runtime activation checks.

## Qwen3.6 MTP Reference Facts

The JANG-side verified Qwen3.6 27B MTP reference bundles used for Swift
runtime work are:

- `/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP`
- `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP`
- `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP8-MTP`
- `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP`

Use the 27B JANG_4M, 27B MXFP4, and 27B MXFP8 bundles as the first live
probes: JANG_4M proves the JANG affine/mixed format path, MXFP4 proves the
native MXFP4 path, and MXFP8 proves the true MXFP8/no-bias path. Use the 35B
A3B MXFP8 bundle for MoE/VL native-MTP module-layout and speed work. Do not
substitute CRACK artifacts when testing MTP; the CRACK variants intentionally
do not carry MTP tensors.

The verified JANG_4M MTP bundle has these properties:

- 29 indexed shards.
- `runtime.total_weight_bytes=17820460160`
- `runtime.total_weight_gb=16.6`
- `runtime.mtp_mode=preserved_enabled`
- 31 converted `mtp.*` tensor entries.
- 333 `vision_tower.*` tensor entries.
- `preprocessor_config.json` and `video_preprocessor_config.json` present.
- JANG Python probe loaded it with `Qwen3VLProcessor`.
- Text probe answered `2 + 2` as `4`.
- Image probe on a generated red square answered `red`.

The copied 35B A3B MXFP8-MTP bundle has these properties:

- local path: `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP`
- copied from:
  `erics-m5-max.local:/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP`
- 47 local regular files, 35G on disk.
- `model_type=qwen3_5_moe`
- `architectures=["Qwen3_5MoeForConditionalGeneration"]`
- `runtime.total_weight_bytes=37530167888`
- `runtime.total_weight_gb=34.95`
- `runtime.mtp_mode=preserved_enabled`
- 31 MTP tensor entries and 333 vision tensor entries.
- `quantization.mode=mxfp8`, `group_size=32`, `bits=8`
- `quantization.norm_convention=qwen3_5_language_mlx_plus_one`
- no MTP `.biases`; MTP linears use MXFP8 weights/scales plus fp16 norms.

That proves the artifact preserves MTP and VL. Swift now distinguishes real
tensor-proven runtime eligibility from metadata-only claims:

```text
mode=preserved_enabled
hasCompleteMTPArtifact=true
hasUsableNativeMTPTuning=true when vmlx_mtp_tuning.json validates the artifact
speculativeDecodeEnabled=true only after tensor evidence plus usable tuning
canAutoLaunchMTP=true only after tensor evidence plus usable tuning
requiresAcceptRejectBeforeEnable=false
```

## Detection Rules

`MTPBundleInspector` detects configured MTP layers from:

- `num_nextn_predict_layers`
- `mtp_num_hidden_layers`
- `text_config.num_nextn_predict_layers`
- `text_config.mtp_num_hidden_layers`
- `jang_config.runtime.mtp_layers`

It detects MTP tensors from:

- top-level `mtp.*`
- `model.mtp_layers.*`
- names containing `.mtp.` or `.mtp_layers.`
- `nextn` / `next_n` names
- layer-N MTP layouts such as `model.layers.<num_hidden_layers>.*`
  and `language_model.model.layers.<num_hidden_layers>.*`

Directory names are not MTP evidence. `jang_config.runtime.bundle_has_mtp`,
`mtp_layers`, or `mtp_mode` can establish that the bundle expected MTP, but
`MTPBundleStatus.bundleHasMTP` is true only when tensor names prove that MTP
weights are present. Metadata without tensor evidence is reported as
`metadata_only_missing_weights`.

For Qwen native MTP, tensor evidence is still not enough to auto-launch. The
inspector also reads `vmlx_mtp_tuning.json` from the bundle root. Auto launch is
allowed only when that file's `native_mtp` row is validated, output-equivalent,
unblocked, and carries a positive `best_depth`. Missing or blocked tuning keeps
`canAutoLaunchMTP=false`, even if the MTP tensors are present.

It detects VL tensors separately from:

- `vision_tower.*`
- `visual.*`
- `vision_model.*`
- `multi_modal_projector.*`
- `mm_projector.*`
- `image_newline*`

This gives Osaurus one unified status object for text-only MTP, VL+MTP, and
metadata-only bundles.

## Osaurus Status Surface

Osaurus should read `context.configuration.mtpStatus` after the model context is
loaded and serialize `context.configuration.mtpStatus?.snapshot`. The raw
`MTPBundleStatus` remains `Codable`, but `snapshot` is the stable status payload
that includes computed gates such as `can_auto_launch=false` for missing or
blocked tuning. Suggested capability/health shape:

```json
{
  "mtp": {
    "mode": "preserved_enabled",
    "bundle_has_mtp": true,
    "configured_layers": 1,
    "tensor_count": 31,
    "vision_tensor_count": 333,
    "has_complete_artifact": true,
    "has_usable_native_mtp_tuning": true,
    "speculative_decode_enabled": true,
    "can_auto_launch": true,
    "requires_accept_reject_before_enable": false,
    "status_line": "mtp: preserved_enabled, layers=1, tensors=31, tuning=d3, speculative=on",
    "tuning": {
      "file": "vmlx_mtp_tuning.json",
      "best_depth": 3,
      "validated": true,
      "output_equivalent": true,
      "blocked": false
    }
  }
}
```

Osaurus launch policy must be:

- `mode=none`: run normal autoregressive decode; report MTP unavailable.
- `mode=metadata_only_missing_weights`: run normal autoregressive decode; report
  that config advertises MTP but the artifact is missing MTP tensors.
- `mode=preserved_disabled`: run normal autoregressive decode; report MTP
  preserved but disabled.
- `mode=preserved_enabled`, `enabled`, or `speculative_verified`: call the
  full-evidence resolver. Supported Qwen profiles with real MTP tensors and a
  usable `vmlx_mtp_tuning.json` row auto-launch native MTP at that row's
  `best_depth`; unsupported, missing-tuning, or blocked profiles remain off or
  blocked if force-on is requested.

If a caller explicitly requests MTP while the full-evidence resolver returns
`.off` or `.blocked`, Osaurus should return a clear unsupported/error response.
It must not silently route through a fake guard, force a hidden sampler
fallback, cap output length, or pretend speculative MTP ran.

Swift now exposes a full-evidence settings bridge for this policy:

- `VMLXServerRuntimeSettings.resolvedMTPLaunch(configData:jangConfig:status:)`
  combines server settings, raw `config.json`, optional `jang_config.json`, and
  `MTPBundleStatus`. It blocks profiles with generic `speculative_verified`
  status but no supported native-MTP recommendation, such as Qwen3.6 JANG_2K,
  and blocks tensor-proven Qwen bundles when `vmlx_mtp_tuning.json` is missing
  or unusable.
- `VMLXServerRuntimeSettings.resolvedMTPDraftStrategy(...)` returns
  `.nativeMTP(depth:)` only when the full-evidence launch decision is
  `.speculative`.
- `VMLXServerRuntimeSettings.resolvedLoadConfiguration(...)` returns a
  `LoadConfiguration` with `nativeMTP=true` for the same full-evidence
  `.speculative` decision. Osaurus should use this load configuration and the
  draft strategy together so the sidecar weights are present before the request
  selects native MTP.
- `mtp.draftTokenLimit` must be positive when set. If it is below the
  recommended depth, the resolver caps the native-MTP depth and records
  `server_draft_token_limit=<n>` in the recommendation evidence.
- `effectiveMTPLaunchMode(for:)` is still a cheap display helper, but it now
  requires usable tuning before reporting `.speculative`. Osaurus should still
  use the full-evidence resolver before launch because model-type and
  profile-specific blocks require `config.json` and optional `jang_config.json`.

## Explicit Swift Runtime Activation

The package has an automatic Qwen3.6 native-MTP path for supported bundles with
real MTP tensor evidence and a usable bundle-local `vmlx_mtp_tuning.json`. The
concrete runtime still uses `LoadConfiguration.nativeMTP=true` plus
`DraftStrategy.nativeMTP(depth:)`; the server settings resolver now derives both
from the same evidence snapshot.
`ModelFactory.loadModel` passes that load-time decision through a task-local
activation override so concurrent loads do not share a process-global MTP env
flag. The `VMLINUX_NATIVE_MTP=1` environment knob remains compatibility-only
for direct factory callers that bypass `ModelFactory.loadModel`; Osaurus should
use the typed load configuration. Activation is not inferred from model names
and is not inferred from `mtp_num_hidden_layers` alone. It requires:

- supported Qwen3.5/Qwen3.6 text, VL, or Qwen3.5 MoE model type;
- complete MTP tensor evidence from the index or safetensors headers;
- usable `vmlx_mtp_tuning.json` metadata for the exact bundle;
- an active Swift model exposing `NativeMTPModel`;
- resolved load-time and request-time runtime selection from
  `VMLXServerRuntimeSettings`;
- greedy `temperature=0` or the native exact p/q stochastic verifier path
  described below.

Bundles whose config advertises MTP but whose weights do not contain MTP tensors
fail closed. The local CRACK bundle
`/Users/eric/models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK` currently reports:

```text
native MTP was requested but this bundle does not have complete MTP tensor evidence:
mtp: metadata_only_missing_weights, layers=1, tensors=0, speculative=off
```

The implementation uses private MTP draft cache and target-model verification.
For Qwen3.6 hybrid SSM/KV, the verifier records accepted-prefix Mamba state and
trims rejected attention-KV suffixes, so rejected draft state is not kept in the
backbone cache. Partial rejects recreate the private MTP draft cache instead of
trimming stale rejected state. This is real D3 prefix-commit semantics, not an
all-or-nothing guard.

Native MTP now accepts a `CacheCoordinator` through both public
`Evaluate.generate(...)` and `BatchEngine.generate(...)` exclusive solo paths.
For Qwen3.6 hybrid/Mamba cache topology the correct cache route is
`pagedIncompatible=true`: generic paged KV counters stay zero by design, and
the restorable store is disk L2 plus SSM companion state. Osaurus should show
this as "Block/L2 + SSM companion active" rather than "paged cache failed."
The current repeated exact-prompt rows prove store/fetch counters and safe
restoration, not a native-MTP TTFT compute-reuse win: path-dependent exact full
hits intentionally reset to full prefill to avoid double-counting recurrent SSM
state. A growing-chat partial-hit native-MTP row is still required before
claiming prompt-compute reuse for this topology.

Qwen3.6 35B A3B MXFP8-MTP is allowed through the explicit activation check only
because its config reports `qwen3_5_moe` and its index has real MTP tensor
evidence. This is not name-based activation. Text MTP D3 load/generate/speed and
disk+SSM cache rows now pass in Swift; VL multi-turn with native MTP is still a
separate row and must not be inferred from text proof.

## Current Swift Live Proof - 2026-05-17

Artifacts are under
`docs/local/production-readiness/20260517Tswift-mtp-current/`.

| Bundle | Row | Artifact | Result |
| --- | --- | --- | --- |
| 27B JANG_4M MTP | D3 count prompt | `qwen36_27b_jang4m_mtp_d3_count_python_prompt_normfix_regression.log` | coherent, `47.7 tok/s`, `verifyCalls=54`, `acceptedByDepth=0:1,1:5,2:12,3:36`, no loop/leak |
| 27B MXFP4 MTP | D3 count prompt | `qwen36_27b_mxfp4_mtp_d3_count_python_prompt_normfix.log` | coherent, `50.5 tok/s`, `verifyCalls=57`, `acceptedByDepth=0:3,1:8,2:12,3:34`, no loop/leak |
| 27B MXFP8 MTP | D3 count prompt | `qwen36_27b_mxfp8_mtp_d3_count_python_prompt_normfix.log` | coherent, `29.5 tok/s`, accepts depths 1/2/3 after independent MTP norm shift; speed still below Python reference |
| 35B A3B MXFP8 MTP | D3 count prompt | `qwen36_35b_mxfp8_mtp_d3_count_python_prompt.log` | coherent, `130.6 tok/s`, `verifyCalls=48`, `acceptedByDepth=2:2,3:46`, no loop/leak |
| 27B MXFP4 MTP | hybrid disk cache repeated prompt | `qwen36_27b_mxfp4_mtp_d3_hybrid_disk_cache_repeated_prompt.log` | run1 disk hit increments to `1`, SSM hit increments to `1`, paged remains zero with `pagedIncompatible=true` |
| 27B MXFP8 MTP | hybrid disk cache repeated prompt | `qwen36_27b_mxfp8_mtp_d3_hybrid_disk_cache_repeated_prompt.log` | run1 disk hit increments to `1`, SSM hit increments to `1`, coherent output |
| 35B A3B MXFP8 MTP | hybrid disk cache repeated prompt | `qwen36_35b_mxfp8_mtp_d3_hybrid_disk_cache_repeated_prompt.log` | run1 disk hit increments to `1`, SSM hit increments to `1`, coherent output |
| 27B MXFP4 MTP | post-audit D3 count prompt | `qwen36_27b_mxfp4_mtp_d3_count_python_prompt_postaudit_count.log` | task-local load activation with process env unset, coherent `1..50`, `49.6 tok/s`, `verifyCalls=56`, `acceptedByDepth=0:5,1:5,2:11,3:35`, `stop=stop`, no loop/leak |
| 27B MXFP4 MTP | post-audit hybrid disk cache repeated prompt | `qwen36_27b_mxfp4_mtp_d3_hybrid_disk_cache_postaudit.log` | run0 stores disk entries, run1 `disk hits=1`, `ssm hits=1`, `pagedIncompatible=true`, coherent `1..50`, no loop/leak |

Cache-proof caveat: the three hybrid disk-cache rows above are exact repeated
prompt probes. They prove the coordinator records, fetches, and restores the
right disk+SSM payload and that coherent generation survives the restore path.
They do not prove native-MTP growing-chat partial-hit compute reuse yet.

Pre-fix evidence is intentionally retained:

- `qwen36_27b_mxfp8_mtp_d3_count_python_prompt.log`: MXFP8 MTP accepted zero
  drafts (`acceptedByDepth=0:190`) and ran at `7.0 tok/s`.
- `qwen36_27b_mxfp4_mtp_d3_count_python_prompt_loadmtp.log`: MXFP4 MTP accepted
  zero drafts before independent MTP norm handling.

The root cause was not bad model behavior and not a sampling guard issue. The
repaired MXFP4/MXFP8 language backbones are already MLX-ready, while preserved
MTP norm tensors can still be raw; Swift now shifts MTP norms independently of
the backbone norm convention.

## Clean-Room Runtime Comparison - 2026-05-16

The Swift implementation was checked against the public behavior of two Python
runtime families without copying code:

- `mlx_lm` speculative decode uses separate verifier and draft caches, drafts
  `K` tokens, verifies `[primary, d1, ... dK]` in one target forward, accepts a
  draft prefix, emits the verifier token at the first rejected/bonus position,
  and trims rejected cache suffixes. That matches the Swift greedy structure.
- MTPLX native MTP uses the model's own MTP heads, recursive draft hidden-state
  feedback, an MTP-private cache, target verification over
  `[primary, d1, d2, d3]`, exact probability-ratio acceptance at stochastic
  temperatures, residual correction sampling on rejection, accepted-prefix cache
  commit, and a compiled/tuned small-M verifier hot path.

The important deltas for Swift were:

1. Native MTP draft must return hidden state as well as logits. Swift now does
   this through `NativeMTPForwardResult`.
2. D2/D3 cannot use all-or-nothing cache acceptance. Swift now commits
   `primary + acceptedDraftPrefix` and trims/repairs rejected suffixes.
3. Stochastic MTP cannot compare sampled token IDs. Swift now has
   `SpeculativeSamplingController`, which applies the same top-p, min-p, top-k,
   and temperature distribution shape as the existing sampler, then performs
   `min(1, p/q)` acceptance and residual correction.
4. The target verifier processor state must advance with draft tokens while
   computing target distributions for later draft positions. It must not advance
   with an unrelated target sample before the draft is accepted.
5. The remaining speed delta is not another sampler or fake guard problem. The
   measured wall is still target verification, so the next real speed path is a
   compiled/retuned small-M verifier shape bank plus any required MLX qmv kernel
   work.

This is intentionally a behavioral contract, not a source port. Python project
structure, function names, and implementation text were not copied into Swift.

Live current-code artifacts under `docs/local/native-mtp-qwen36-20260515/`
record the following local rows:

| Bundle | Mode | Artifact | Result |
| --- | --- | --- | --- |
| `Qwen3.6-27B-JANG_4M-MTP` | AR base | `jang4m-mtp-artifact-ar-normdetect-256.log` | coherent, `stop=stop`, `loop=NO`; norm convention detected from weights |
| `Qwen3.6-27B-JANG_4M-MTP` | native MTP D1 | `jang4m-mtp-artifact-native-d1-mtpfcfix-256.log` | coherent, `23.6 tok/s` median, `verifyCalls=106`, `avgCommittedPerVerify=1.62` |
| `Qwen3.6-27B-JANG_4M-MTP` | native MTP D2 | `jang4m-mtp-artifact-native-d2-mtpfcfix-256.log` | coherent, `19.7 tok/s` median, `verifyCalls=85`, `avgCommittedPerVerify=2.02` |
| `Qwen3.6-27B-JANG_4M-MTP` | native MTP D3 | `jang4m-mtp-artifact-native-d3-mtpfcfix-256.log` | coherent, `12.2 tok/s` median, `verifyCalls=85`, `avgCommittedPerVerify=2.11` |
| `Qwen3.6-27B-JANG_4M-MTP` | native MTP D1 post-cleanup | `jang4m-mtp-artifact-native-d1-postrevert-96.log` | coherent, `30.2 tok/s` median on a short 48-token stop, `loop=NO` |
| `Qwen3.6-27B-MXFP4-MTP` | native MTP D1 post-cleanup | `mx-mtp-artifact-native-d1-postrevert-96.log` | coherent, `34.0 tok/s` median on a short 51-token stop, `loop=NO` |
| `Qwen3.6-27B-JANG_4M-CRACK` | native MTP requested | `jang4m-crack-native-mtp-denied-postrevert.log` | fail-closed, `metadata_only_missing_weights`, exit status 133 |

Live 2026-05-16 D3 prefix-commit artifacts under
`docs/local/native-mtp-qwen36-20260516-d3-prefix-commit/`:

| Bundle | Mode | Artifact | Result |
| --- | --- | --- | --- |
| `/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP` | AR baseline | `jang4m-mtp-ar-baseline-256.log` | coherent, `13.9 tok/s`, `stop=length`, `loop=NO` |
| `/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP` | native MTP D3 prefix commit | `jang4m-mtp-d3-prefix-commit-no-checkpoint-256.log` | coherent, `15.0 tok/s`, `verifyCalls=117`, `prefixCommit=117`, `rollbackRepair=0`, `avgCommittedPerVerify=2.19`, `loop=NO` |
| `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP` | AR baseline | `mxfp4-mtp-ar-baseline-256.log` | coherent, `14.6 tok/s`, `stop=length`, `loop=NO` |
| `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP` | native MTP D3 prefix commit | `mxfp4-mtp-d3-prefix-commit-no-checkpoint-256.log` | coherent, `16.4 tok/s`, `verifyCalls=116`, `prefixCommit=116`, `rollbackRepair=0`, `avgCommittedPerVerify=2.21`, `loop=NO` |

Live 2026-05-16 exact p/q stochastic native-MTP artifacts under
`docs/local/native-mtp-qwen36-20260516-cleanroom/`:

| Bundle | Mode | Artifact | Result |
| --- | --- | --- | --- |
| `/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP` | native MTP D3, `temp=0.6`, `top_p=0.95`, `top_k=20` | `jang4m-mtp-d3-exactpq-temp06-64.log` / `.err` | coherent, `stop=stop`, `loop=NO`, `tokps=8.2`, `verifyCalls=17`, `acceptedByDepth=0:5,1:8,2:1,3:3`, `residualCorrection=14`, `avgAcceptP=0.591`, `samplingMode=exact-pq` |
| `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP` | native MTP D3, `temp=0.6`, `top_p=0.95`, `top_k=20` | `mxfp4-mtp-d3-exactpq-temp06-64.log` / `.err` | coherent, `stop=stop`, `loop=NO`, `tokps=27.5`, `verifyCalls=17`, `acceptedByDepth=0:1,1:7,2:4,3:5`, `residualCorrection=12`, `avgAcceptP=0.686`, `samplingMode=exact-pq` |

Phase-timing reruns (`*-phase-256.*`) show the wall is target verify, not cache
commit: JANG_4M spent `16.388s` in target verify, `1.903s` in MTP draft,
`0.140s` in sampling, and `0.088s` in cache commit; MXFP4 spent `15.411s`,
`1.202s`, `0.118s`, and `0.069s` respectively. This points the next speed pass
at compiled/tuned small-M verifier execution, not another cache monkeypatch.

Live 2026-05-16 BatchEngine dispatch artifacts under
`docs/local/live-model-matrix/20260516Tbatch-mtp-dispatch/`:

| Bundle | Mode | Artifact | Result |
| --- | --- | --- | --- |
| `/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP` | `BatchEngine.generate`, native MTP D3 | `Qwen3.6-27B-JANG_4M-MTP_batch_native_mtp_d3.out` / `.err` | coherent short row, path=`batch`, `33.3 tok/s`, `stop=length`, `unclosedReasoning=NO`, `loop=NO`, `leaks=none`, `[NativeMTP] depth=3 verifyCalls=13 prefixCommit=13 rollbackRepair=0 avgCommittedPerVerify=2.46` |

Focused test proof for the same dispatch contract:

- `MTPRuntimeFocusedTests` passes 17/17 after the 2026-05-17 text hybrid-SSM
  offset fix and private MTP-cache reject refresh fix.
- Post-audit rerun
  `docs/local/production-readiness/20260517Tswift-mtp-current/mtp_runtime_focused_postaudit.log`
  passes 22/22 `MTPRuntimeFocusedTests`, including task-local activation,
  MXFP8 norm metadata rows, text-SSM offset capture, private MTP-cache refresh,
  active BatchEngine native-MTP dispatch, missing-head fail-closed dispatch, and
  `BatchEngine.submit` rejection.
- `BatchEngine.generate` with active native MTP reaches the real
  `NativeMTPTokenIterator` and emits native-MTP telemetry.
- `BatchEngine.generate` with a requested native-MTP strategy but no active MTP
  head fails closed instead of running plain AR.
- `BatchEngine.submit` rejects native MTP instead of silently treating it as
  ordinary batched decode. Raw multi-slot native-MTP scheduling is not
  implemented yet.
- `Qwen35GatedDeltaNet` in the text path now matches the VLM path for SSM
  offsets: normal forwards advance `MambaCache.offset`, and verifier
  accepted-prefix snapshots store `baseOffset + prefixLength`. The focused log
  is
  `docs/local/production-readiness/20260517Tqwen36-mtp-ssm-offset/mtp_runtime_focused_after_private_mtp_refresh.log`.
- Partial rejection now recreates the private MTP draft cache before drafting
  from the correction token. It no longer trims the old private cache, so stale
  rejected draft KV/state cannot survive into the next draft round. Telemetry
  reports this as `mtpCacheRefresh`.

Live 2026-05-17 paired JANG_4M text rows after the text-SSM offset fix:

| Bundle | Mode | Artifact | Result |
| --- | --- | --- | --- |
| `/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP` | AR baseline, same prompt | `qwen36_27b_jang4m_ar_text_live.log` | coherent, `20.0 tok/s`, `stop=length`, `loop=NO`, `leaks=none` |
| `/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP` | native MTP D3, same prompt | `qwen36_27b_jang4m_mtp_d3_private_cache_refresh_live.log` | coherent, `18.0 tok/s`, `prefixCommit=50`, `rollbackRepair=0`, `mtpCacheRefresh=46`, `avgCommittedPerVerify=1.92`, `loop=NO`, `leaks=none` |

The D3 row is a correctness/coherency check for text hybrid-SSM partial-prefix
commit. It is not a production speed pass, because it is slower than the
matching AR row on this prompt and remains far below the 45 tok/s threshold.

2026-05-17 MXFP8 norm-convention follow-up:

- `loadWeights` now merges `norm_convention` from safetensor metadata or
  `jang_config.json` into the metadata passed to `LanguageModel.sanitize`.
- Qwen3.5/Qwen3.6 text and VLM sanitizers honor
  `qwen3_5_language_mlx_plus_one` explicitly for language and MTP norms.
- If a bundle explicitly reports a different norm convention, conv1d layout
  sanitization still runs but does not imply a norm shift. This keeps the
  MXFP8 norm contract orthogonal to the conv layout contract.
- Focused proof:
  `docs/local/production-readiness/20260517Tqwen36-mtp-ssm-offset/mtp_runtime_focused_after_norm_convention_v2.log`
  passes 19/19 `MTPRuntimeFocusedTests`. Broader proof:
  `docs/local/production-readiness/20260517Tqwen36-mtp-ssm-offset/mlxlmcommon_focused_after_norm_convention.log`
  passes 85/85.
- This is still unit/static proof in Swift until the local MXFP8 artifacts are
  present and rerun through live text, cache-on, and VL gates.

2026-05-16 Qwen3.6 VLM/MRoPE follow-up:

- `Libraries/MLXVLM/Models/Qwen35.swift` now factors normal Qwen3VL position-ID
  resolution into one helper and uses it from the native-MTP backbone verifier.
  After image or video prefill, verifier and bridge forwards therefore use the
  same `ropeDeltas`/precomputed-position continuation state as normal decode
  instead of falling back to raw text-only cache offsets. This is required for
  2D image and 3D video MRoPE correctness.
- Focused proof:
  `docs/local/live-model-matrix/20260516Tqwen35-vlm-mrope-mtp/mlxlmcommon_focused_after_qwen35_vlm_moe_sidecar.out`
  passes 81/81 tests across 13 suites, including the new
  `Qwen3.6 VLM native MTP verifier reuses MRoPE continuation state` and
  `Qwen3.6 VLM native MTP decoder uses sparse MoE for MoE sidecars` rows plus
  Hadamard 3D/4D, JANGTQ rank-2/rank-3 matmul, media salt, hybrid cache,
  DSV4 cache topology, and no-hidden-guard rows.
- No-load tensor-key census:
  `docs/local/live-model-matrix/20260516Tqwen35-vlm-mrope-mtp/qwen36_mtp_vl_tensor_census.json`
  records `/Users/eric/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP` with 31 MTP
  tensor entries and 333 vision entries, and
  `/Users/eric/models/JANGQ/Qwen3.6-27B-MXFP4-MTP` with 23 MTP entries and 333
  vision entries, and
  `/Users/eric/models/JANGQ/Qwen3.6-35B-A3B-MXFP4-MTP` with 42 MTP entries,
  333 vision entries, `qwen3_5_moe`, and `qwen3_5_moe_text`. The optional
  real-bundle inspector tests also pass for all three paths in the same
  artifact directory.
- 35B transfer proof: `du -sh` reports `22G`, the local folder has 37 regular
  files, and dry-run `rsync --delete --itemize-changes` from
  `erics-m5-max2.local` reported no remaining differences.
- MoE/VL native-MTP readiness proof:
  `docs/local/live-model-matrix/20260516Tqwen35-vlm-mrope-mtp/mlxlmcommon_focused_after_qwen35_vlm_moe_sidecar.out`
  passes 81/81 tests across 13 suites after enabling `qwen3_5_moe` /
  `qwen3_5_moe_text` in the explicit activation allowlist and changing the VLM
  MTP decoder to instantiate `SparseMoeBlock` for MoE sidecars.

The current 45 tok/s acceptance threshold, and the older 50 tok/s Qwen3.6 27B
target, are not achieved by every current Swift row. Native MTP is still
auto-launch limited to supported tensor-proven Qwen bundles, and release review
must keep token/s plus output-tail evidence attached. The attempted verifier
argmax vectorization was rejected after live rows were slower, so it is not part
of the implementation. The next real speed work is:

1. recursive MTP draft returns logits and hidden state for `d1`, `d2`, and
   `d3` without recomputing state traces;
2. one target verifier forward over `[primary, d1, d2, d3]`;
3. compiled/tuned small-M verifier shapes for Qwen3.6 hybrid blocks, with cache
   offsets represented in the graph rather than as Python/Swift scalar state;
4. phase timing for target verify time, MTP draft time, cache commit time, and
   sampling time;
5. telemetry for requested/effective depth, verify calls, accepted-by-depth,
   bonus tokens, correction count, target verify time, MTP draft time, and output
   tail review.

A hidden sampler floor, forced repetition penalty, forced stop, or all-or-
nothing accept rule is not an acceptable substitute.

## Runtime Activation Contract

The activation path is family-specific. Do not add a global auto-MTP switch
until at least one family implements all of:

1. Load the MTP head/layer without breaking the base autoregressive loader.
2. Return both logits and hidden state from each MTP draft step. D2/D3 recursive
   draft cannot be built from a logits-only `mtp_forward`.
3. Keep a temporary draft cache/state separate from accepted base KV.
4. Propose recursive draft tokens up to the requested depth.
5. Verify `[primary, d1, ... dK]` through the base model in one target forward.
6. Commit the primary token plus accepted draft prefix of length `0...K` into
   the base cache stack. A verifier bonus token is emitted as the next primary
   and committed by the following verifier cycle; it is not silently written
   into cache without a target forward. The backbone cache never receives
   rejected draft state.
7. Discard draft state on rejection, cancellation, stop, or request failure.
8. Report verify cycles, accepted/rejected draft counts, acceptance rate,
   fallback count, and token/s.

Depth matters. A D1 loop that drafts one token and verifies two positions is not
the MTPLX-style target. For a 256-token response, D1 still takes about 128
verify cycles at full acceptance. A D3 path verifies `[primary, d1, d2, d3]` and
can commit up to four tokens per cycle, so the full-acceptance lower bound is 64
verify cycles, with real rows expected around 50-70 depending on bonus handling
and stop behavior.

The Swift contract type for this is `MTPRecursiveDraftContract`. Its D3 shape
requires hidden-state draft feedback, private draft cache, accepted-only
backbone commit, variable `0...depth` accepted-prefix commit, and a compiled or
tuned small-M verifier hot path before any speed claim is accepted.

Current runtime state: D3 MLLM native-MTP has correct cache boundaries for the
Qwen3.6 text path, including text hybrid-SSM accepted-prefix offsets during
partial rejection, the Qwen3VL native-MTP backbone verifier now shares the
normal VLM MRoPE continuation-state resolver, and the VLM MTP decoder chooses
sparse MoE layers for Qwen3.6 35B-style MoE sidecars. `Evaluate.generate` and
`BatchEngine.generate` both honor an explicit `.nativeMTP(depth:)` request. The
BatchEngine path is deliberately an exclusive solo lane; it is not a multi-slot
paged native-MTP scheduler. `BatchEngine.submit` rejects native MTP so a raw
batched caller cannot get a fake AR pass while believing MTP ran.

Prefix, paged KV, and block-L2 disk remain prompt-boundary verified caches.
Each D3 verify pass advances the live backbone cache through verified target
positions only; rejected draft suffixes are not kept. The remaining production
blockers are speed and composition: prefix commit now avoids full
rollback+repair, but the verifier/prefix-state hot path is not tuned enough to
beat the 45 tok/s threshold, and native MTP has not been live-proven with real
image/video multi-turn media payloads, paged KV/block-L2, or hybrid SSM
companion-cache rederive.

Qwen-style bundles use top-level `mtp.fc.*` and `mtp.layers.0.*` tensors. Hy3
and Bailing-style bundles may store the MTP layer at
`model.layers.<num_hidden_layers>.*`. Those paths need separate adapters.

## Cache Rules

The base cache stack is authoritative:

- Prefix cache, paged cache, disk L2 cache, media cache, and SSM companion cache
  may contain accepted base-model state only.
- Draft MTP KV/state is temporary. It must not be written into prefix, paged, or
  disk L2 cache until the base model accepts the token.
- Rejected drafts must trim/discard draft state without mutating accepted base
  state.
- Mid-stream cancellation must leave no in-flight draft state behind.

Cache-key policy:

- Status-only `preserved_enabled` does not change the plain autoregressive cache
  key, because the generation path is still base decode.
- An actual MTP-enabled generation path must salt cache keys with at least:
  model revision, quant profile, tokenizer/chat-template salt, parser mode,
  `mtp_mode`, MTP family adapter, draft depth, and media salt when media exists.
- Text-only VL turns must keep media salt nil, exactly like the existing VL cache
  contract.
- A turn with new image/video/audio input must not reuse draft state from a prior
  media salt.

Hybrid/SSM rule:

- MTP draft recurrent state is not an SSM companion cache entry.
- Async re-derive for hybrid models must re-derive from accepted base tokens and
  accepted companion state only.
- For D2/D3, rollback/commit must support accepted draft prefix length `0...K`;
  a single accept/reject bit is not enough for hybrid SSM correctness.
- Accepted: keep the verifier state after `[primary, accepted drafts...]`.
- Rejected: restore or repair to the state after the accepted prefix, not
  blindly to `primary` and not past the rejected draft.

For D2/D3 this partial-acceptance rule is mandatory. If the verifier accepts
`d1` and `d2` but rejects `d3`, the backbone cache must commit state after
`[primary, d1, d2]`. It must not roll all the way back to `primary`, and it
must not keep the rejected `d3`.

There are two correct implementation options:

1. Capture/commit path: record intermediate hybrid SSM/KV states during the
   verifier forward and commit the selected accepted prefix.
2. Rollback + repair path: rollback to `primary`, then re-forward the accepted
   prefix plus correction through the target model.

The capture/commit path is the speed path. The rollback+repair path can be a
correctness-first stepping stone, but it will reduce speed whenever partial
rejections occur. It must still be explicit and measured; hidden guards or
all-or-nothing acceptance are not acceptable production semantics.

## Speed Bench Requirements

Every future native-MTP speed claim must report:

- AR baseline tok/s and MTP tok/s on the same artifact, machine, sampler, and
  prompt set;
- MTP depth requested and effective;
- verify calls;
- output tokens;
- accepted/drafted by depth;
- average committed tokens per verify call;
- bonus-token count;
- correction/rejection count;
- target verify forward time;
- MTP draft time;
- accept/residual sampling time;
- cache mode (`off`, `paged+ssm`, etc.);
- whether small-M compiled verify or stock MLX verify was used;
- whether a draft-only LM head or MTP sidecar was used; and
- output tail review.

Without those fields, a tok/s number is not diagnosable.

## VL Rules

VL+MTP bundles need both sides proven:

- MTP status must show complete MTP artifact state.
- VL status must show vision tensors and processor metadata.
- Turn 1 image+text, turn 2 text-only, turn 3 different image must preserve the
  current media-salt behavior.
- MTP draft state must be scoped under the same media salt as the base verifier.
- Video processors must be checked separately from still images because the
  frame/time axes affect prompt placeholders and cache salts.

The Qwen3.6 JANG_4M MTP reference bundle is a VL+MTP artifact. Swift support is
not complete until Qwen3VL text, image, video, multi-turn cache, and MTP on/off
rows all pass with coherent output. The current Swift source path now preserves
Qwen3VL MRoPE continuation state during native-MTP verifier forwards, but this
is still only a prerequisite for the live VL+MTP gate, not the gate itself.

## Verification Gates For Auto-Launch Expansion

Before changing native-MTP auto-launch beyond supported Qwen MXFP/JANG_4M
profiles, produce artifacts for:

- No-load status: `MTPBundleInspector` sees config layers, MTP tensors, and VL
  tensors correctly.
- Load: the model loads with the base path and does not materialize MTP-only
  state into normal decode.
- MTP off: multi-turn output remains coherent and cache counters match the base
  path.
- MTP on: coherent multi-turn output, normal stop, no looping, no hidden
  reasoning-only output, accepted/rejected draft counters, token/s, and low
  physical footprint where relevant.
- Cache: prefix, paged, disk L2, media, and SSM companion behavior remain correct
  with MTP enabled.
- VL: image+text, text-only resume, and different-image turns remain grounded in
  the right media.
- Inverse: disabling MTP restores exact plain autoregressive launch behavior.

There is no package-level claim that "MTP works" until a named model family
passes those rows. The current state is truthful auto-detection and Osaurus
status propagation.
