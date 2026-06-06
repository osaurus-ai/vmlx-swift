# Distributed Inference Roadmap

Status date: 2026-05-07.

This repo has the foundations for distributed inference, but production
serving remains local by default. Distributed paths must be explicit because
cache format, model topology, and failure semantics differ by architecture.

## Current State

- `MLXDistributedCore` owns peer identity, TXT schema, discovery hooks,
  planning, and local fallback.
- `Mode.replica` is request-level fan-out through an injected transport. It
  now plans only against peers that advertise replica mode, the requested
  model hash or an overflow model list, and a TLS endpoint.
- `Mode.pipelined` is a two-rank TLS prompt/token path. It now plans only
  against peers that advertise pipelined mode, the requested model hash or
  an overflow model list, and a TLS endpoint.
- `MLXDistributedTransport` contains frame encoding, cert/trust helpers,
  stage client/server, stage handler, and `TLSPipelinedTransport`.
- `MLXDistributedTP` contains single-rank-safe collectives and sharded
  linear wrappers over the MLX distributed C ABI shim.
- `MLXDistributedJACCL` currently exposes backend availability checks. Full
  multi-rank JACCL collectives are not production-wired here.
- `Tools/tp-launch-2host.sh` is generic and environment-driven. It has no
  baked-in retired host defaults and should only be run after both hosts are
  explicitly configured.

## Runtime And Cache Matrix

| Family | Distributed classification | Cache requirements |
| --- | --- | --- |
| Qwen, Gemma, Laguna, Nemotron text | Standard transformer KV/MoE | Paged prefix, disk L2, TurboQuant KV, and continuous batching can be shared only after exact model hash and tokenizer/template identity match. |
| MiniMax M2/M2.7 | Standard KV/MoE transformer, not hybrid | Use normal KV cache, paged prefix, disk L2, TurboQuant KV, and batching. Do not route it through SSM/CCA/DSV4 hybrid rollback policy. |
| Ling/Bailing | Recurrent linear-attention hybrid | ArraysCache plus recurrent state must move together. Prefix hits and prompt-boundary rederive need exact recurrent state restore; do not treat it as plain KV. |
| ZAYA | CCA hybrid | `ZayaCCACache` contains KV, convolution state, and previous hidden state. Paged prefix remains disabled for this family until CCA block identity and restore semantics are proven. Disk L2 exact round-trip is the current supported persistence path. |
| DSV4 Flash | CSA/HSA/SWA attention mix | Cache policy must preserve each attention class separately. Do not force TurboQuant KV over CSA/HSA/SWA compression paths until the DSV4 cache contract is encoded in the planner. |
| Omni/VL families | Text KV plus media-derived context | Cache keys must include media salt and modality metadata. Image, audio, and video turns cannot share text-only prefix entries unless media state is identical. |

## Build Phases

1. **Planner correctness**: keep `.auto` local, require explicit replica or
   pipelined mode, filter peers by mode/model/TLS, preserve stage ordering.
   This is the current slice.
2. **Loopback transport proof**: keep TLS loopback tests and add a real
   `ClusterSession` integration row for request end and error propagation.
3. **Stage execution contract**: replace prompt/token echo with typed
   activation envelopes, layer-range metadata, dtype, shape, tokenizer
   identity, cache identity, and end-of-response reasons.
4. **TP family plans**: start with dense Llama/Gemma-style layers, then
   Qwen/Gemma MoE. JANGTQ/MXFP4 TP follows only after packed-weight shard
   contracts are explicit.
5. **Cache-aware distributed admission**: block cross-peer prefix hits unless
   model hash, tokenizer/template hash, reasoning mode, tool mode, media salt,
   and family-specific cache state all match.
6. **JACCL/ring multi-rank validation**: prove collectives with size > 1,
   then wire failure, timeout, and replan behavior. Local single-rank tests
   are not enough for production status.
7. **Osaurus integration**: expose placement, peer health, cache eligibility,
   and per-stage tok/s to the host without app-layer model-family guesses.

## Verification Commands

```sh
swift build -c release \
  --target MLXDistributedCore \
  --target MLXDistributedTransport \
  --target MLXDistributedTP \
  --target MLXDistributedJACCL \
  --product TPRankWorker

swift test --filter "ClusterSessionTests|ClusterSessionPipelinedTests|GroupTests|CollectivesSingleRankTests|LinearLayersSingleRankTests|ShardingHelperTests|JACCLAvailabilityTests"
```

Model speed and cache claims still require `RunBench` or a current log from a
real model load. Distributed readiness must be reported separately from normal
local runtime readiness.
