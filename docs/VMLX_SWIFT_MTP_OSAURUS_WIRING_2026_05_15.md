# vMLX Swift MTP / Osaurus Wiring Plan - 2026-05-15

This document records the Swift-side MTP status contract added for Osaurus.
It is intentionally a status and wiring contract, not a claim that speculative
MTP decode is production-ready.

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
autoregressive decode remains the default path unless an explicit future MTP
runtime proves verified accept/reject.

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

That proves the artifact preserves MTP and VL. It does not prove Swift
speculative MTP accept/reject decode, so Swift must report:

```text
mode=preserved_enabled
hasCompleteMTPArtifact=true
speculativeDecodeEnabled=false
canAutoLaunchMTP=false
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

## Future Swift Runtime Activation

The future activation path should be family-specific. Do not add a global
`DraftStrategy.mtp` switch until at least one family implements all of:

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
