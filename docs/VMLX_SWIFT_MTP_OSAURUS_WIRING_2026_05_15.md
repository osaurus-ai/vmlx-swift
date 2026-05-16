# vMLX Swift MTP / Osaurus Wiring Plan - 2026-05-15

This document records the Swift-side MTP status and activation contract for
Osaurus. Native MTP now exists as an explicit, tensor-gated runtime path for
Qwen3.6, but it is still not an automatic production launch mode. Auto-launch
requires the cache, VL, multi-turn, and speed gates below.

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

The JANG-side verified Qwen3.6 27B JANG_4M MTP bundle has these properties:

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

That proves the artifact preserves MTP and VL. Swift must still distinguish
artifact preservation from runtime readiness:

```text
mode=preserved_enabled
hasCompleteMTPArtifact=true
speculativeDecodeEnabled=false unless explicitly requested
canAutoLaunchMTP=false until the full gate passes
requiresAcceptRejectBeforeEnable=true
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
loaded. Suggested capability/health shape:

```json
{
  "mtp": {
    "mode": "preserved_enabled",
    "bundle_has_mtp": true,
    "configured_layers": 1,
    "tensor_count": 31,
    "vision_tensor_count": 333,
    "has_complete_artifact": true,
    "speculative_decode_enabled": false,
    "can_auto_launch": false,
    "requires_accept_reject_before_enable": true,
    "status_line": "mtp: preserved_enabled, layers=1, tensors=31, speculative=off (accept/reject required)"
  }
}
```

Osaurus launch policy must be:

- `mode=none`: run normal autoregressive decode; report MTP unavailable.
- `mode=metadata_only_missing_weights`: run normal autoregressive decode; report
  that config advertises MTP but the artifact is missing MTP tensors.
- `mode=preserved_disabled` or `preserved_enabled`: run normal autoregressive
  decode; report that MTP is preserved but runtime activation is pending.
- `mode=enabled` or `speculative_verified`: MTP may auto-launch only when
  `canAutoLaunchMTP=true`.

If a caller explicitly requests MTP while `canAutoLaunchMTP=false`, Osaurus
should return a clear unsupported/error response. It must not silently route
through a fake guard, force a hidden sampler fallback, cap output length, or
pretend speculative MTP ran.

## Explicit Swift Runtime Activation

The package has an opt-in Qwen3.6 native-MTP path behind
`VMLINUX_NATIVE_MTP=1` plus `DraftStrategy.nativeMTP(depth:)`. Activation is
not inferred from model names and is not inferred from `mtp_num_hidden_layers`
alone. It requires:

- supported Qwen3.5/Qwen3.6 text or VL model type;
- complete MTP tensor evidence from the index or safetensors headers;
- an active Swift model exposing `NativeMTPModel`;
- explicit runtime selection;
- greedy `temperature=0` for the current implementation.

Bundles whose config advertises MTP but whose weights do not contain MTP tensors
fail closed. The local CRACK bundle
`~/models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK` currently reports:

```text
native MTP was requested but this bundle does not have complete MTP tensor evidence:
mtp: metadata_only_missing_weights, layers=1, tensors=0, speculative=off
```

The implementation uses private MTP draft cache, target-model verification, and
explicit rollback+repair on partial rejection. Rejected draft state is not kept
in the backbone cache. This is a correctness-first path, not the speed path.
Depth values greater than one currently exercise recursive draft calls and
partial-accept repair, but they are not production depth-3 acceleration because
the verifier cache still has to be repaired by re-forwarding accepted prefixes
instead of committing an intermediate captured verifier state.

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

The 50 tok/s Qwen3.6 27B target is not achieved by the current Swift path.
The attempted verifier argmax vectorization was rejected after live rows were
slower, so it is not part of the implementation. The next real speed work is
proper depth-3 activation:

1. recursive MTP draft returns logits and hidden state for `d1`, `d2`, and
   `d3`;
2. one target verifier forward over `[primary, d1, d2, d3]`;
3. capture/commit of intermediate Qwen hybrid SSM/KV states for accepted prefix
   length `0...3`;
4. correction-token repair only for the rejected suffix, not a full all-or-
   nothing rollback;
5. compiled or tuned small-M verifier shapes;
6. telemetry for requested/effective depth, verify calls, accepted-by-depth,
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
6. Commit an accepted draft prefix of length `0...K` plus the verifier bonus
   token into the base cache stack. The backbone cache never receives rejected
   draft state.
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

Implementation target for the next runtime pass: D3 MLLM native-MTP with correct
cache boundaries. Prefix, paged KV, and block-L2 disk remain prompt-boundary
verified caches. Each D3 verify pass advances the live backbone cache only
through verified target positions. Until the capture/commit path exists, partial
rejects must use an explicit rollback+repair path so rejected draft state is not
stored. This is acceptable as a correctness-first stepping stone only if the
bench reports coherency, token/s, verify calls, accepted-prefix length, and the
rollback+repair cost.

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
rows all pass with coherent output.

## Verification Gates Before Auto-Launch

Before changing `canAutoLaunchMTP` to true for any family, produce artifacts for:

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
