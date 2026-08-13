# Qwen 3.5 TB5/RDMA Tensor-Parallel Proof Plan

Last updated: 2026-06-10

## Purpose

This note defines the first vMLX-side proof lane for Qwen 3.5 distributed
tensor-parallel inference over a Thunderbolt data plane. It is intentionally
separate from active Gemma 4 release work.

## Current Test Entry Point

```sh
QWEN35_TP_MODEL=/path/to/qwen35/model \
QWEN35_TP_ARTIFACT_DIR=.artifacts/qwen35-tb5-tp-live \
scripts/vmlx-qwen35-tb5-tp-proof.sh
```

Without `QWEN35_TP_MODEL`, the suite still builds the distributed tools and
runs the encrypted peer smoke plus single-rank collective/kernel smoke. That
is only `PARTIAL_NO_MODEL`.

## What The Suite Exercises

- builds `Qwen35TPProofRunner` and `DistributedPeerSmoke`
- verifies the encrypted distributed metadata path
- runs an MLX collective smoke path through the Swift distributed wrapper
- loads a Qwen 3.5 model when `QWEN35_TP_MODEL` is set
- runs deterministic decode with:
  - Qwen 3.5 sharding plan selected
  - prefix cache coordinator enabled
  - disk L2 cache enabled
  - TurboQuant KV mode enabled
- repeats the same prompt as a warm replay and requires disk L2 hit evidence
- writes a machine-readable `SUMMARY.json`

## Qwen 3.5 Architecture Boundary

The first sharding plan covers Qwen-style normal attention and MoE/MLP
projection modules:

- `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj`
- dense `mlp.gate_proj`, `up_proj`, `down_proj`
- shared expert projections

Qwen 3.5 GatedDelta / SSM companion layers are deliberately left replicated.
They need a separate proof for recurrent state, SSM companion cache restore,
prefix replay, and L2 disk compatibility before being called TP-ready.
Routed `switch_mlp` experts are also left replicated in this clean checkpoint
until SwitchLinear sharding support is proven in the same branch.

## Required Artifacts For A Real TB5 RDMA Row

The simulated local row is not enough for release. A real row needs:

- two Macs with Osaurus/vMLX running the same commit
- Thunderbolt data-plane IPs accepted by `TensorDataPlanePolicy`
- no Tailscale or VPN address in the tensor data plane
- `QWEN35_TP_BACKEND=jaccl`
- rank 0 and rank 1 logs
- model load success on all ranks
- nonzero sharding rewrites
- visible decoded output from rank 0
- token/s or equivalent generation timing
- prefix cache evidence
- disk L2 hit/store evidence
- Qwen hybrid SSM companion cache or explicit replicated-companion boundary
- no native crash or hanging collective

## Status

- `FIXED`: Qwen-family sharding plan exists for attention and MoE/MLP
  projection modules.
- `FIXED`: proof runner captures build, peer smoke, collective smoke,
  model decode, warm replay, TurboQuant KV mode, and disk L2 stats.
- `PARTIAL`: local single-host run simulates the distributed path but does not
  prove TB5 RDMA transport.
- `PARTIAL`: Qwen 3.5 GatedDelta/SSM layers and routed SwitchGLU experts are
  replicated pending companion cache and expert-sharding parity proof.
- `BLOCKED`: real multi-node RDMA row until two Thunderbolt-connected Macs are
  available with JACCL/RDMA initialized and matching model bundles.
