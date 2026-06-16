# AdLab RDMA/TB5 Tensor Parallel Swift Engine Import

Updated: 2026-06-10

## Goal

Bring the AdLab RDMA/Thunderbolt tensor-parallel work into vMLX as a Swift
engine capability, not as a Python-side release path. Python files from AdLab
are provenance and behavior references. The target product is Swift package
code, Swift tools, Swift tests, and Swift runtime diagnostics.

## Import Policy

- Prefer additive Swift surfaces first because the current vMLX worktree already
  has active uncommitted TP, MiMo, cache, and distributed edits.
- Do not copy AdLab patch scripts directly into the engine as the implementation.
  Convert the proven behavior into Swift source, probes, and tests.
- Keep AdLab cluster scripts as reference unless they describe a reusable local
  invariant such as route selection, hostfile safety, JACCL backend discovery, or
  metallib kernel packaging.
- Treat every readiness claim as one of `FIXED`, `PARTIAL`, `BLOCKED`, or `TODO`.
  A route pass, size-1 distributed fallback, or `librdma` load is not a TP proof.

## Discovery Stages

### 1. Fabric Discovery

Source references:

- `/Users/eric/adlab/docs/adlab-thunderbolt-networking.md`
- `/Users/eric/adlab/docs/tp4-rdma-prep.md`
- `/Users/eric/adlab/scripts/adlab-tb-net-verify.sh`
- `/Users/eric/adlab/scripts/adlab-rdma-jaccl-preflight.sh`
- `/Users/eric/adlab/scripts/adlab-pod-rdma-preflight.sh`
- `/Users/eric/adlab/scripts/adlab-ibv-roundtrip-pair.sh`

Swift engine import targets:

- `tools/DistributedProbe`
- `Libraries/MLXDistributedCore/JACCL-RDMA-DISCOVERY-BRINGUP.md`
- future Swift diagnostics under `MLXDistributedCore` or `MLXDistributedJACCL`

Required Swift behavior:

- Report Thunderbolt candidate interfaces and route findings.
- Reject tensor-parallel data-plane hostfiles or coordinators that use Tailscale
  `100.x` addresses.
- Distinguish `librdmaLoadable`, `JACCL.isAvailable`, and true multi-rank
  collective readiness.
- Validate `MLX_IBV_DEVICES` shape against world size and rank, including empty
  self slots and non-empty peer entries.
- Never treat size-1 `Group` fallback as distributed proof.

Current status: `PARTIAL`.

vMLX already has `DistributedProbe`, `JACCL.isAvailable`, `librdmaLoadable`, and
basic TXT preview safety. The Swift data-plane classifier now rejects Tailscale
`100.x` and accepts AdLab Thunderbolt `10.20.0.x` / `10.10.x.x` addresses. It
does not yet have full route-table validation, hostfile parsing, IBV-device
matrix proof, or multi-rank proof gates in Swift.

### 2. Swift Runtime Discovery

Source references:

- `/Users/eric/adlab/scripts/adlab-vmlx-release-metallib.sh`
- `/Users/eric/adlab/scripts/adlab-mlx-jaccl-smoke.py`
- `/Users/eric/adlab/vendor/external-rdma-examples/`

Swift engine import targets:

- `Libraries/MLXDistributedJACCL/JACCL.swift`
- `Libraries/MLXDistributedTP/Group.swift`
- `Libraries/MLXDistributedTP/Collectives.swift`
- `tools/TPRankWorker`
- `tools/DistributedProbe`

Required Swift behavior:

- Add typed diagnostics for JACCL backend state instead of relying only on fatal
  collective preconditions.
- Surface collective failures as proof artifacts where possible.
- Add a metallib fast-sync guard for `input_coherent`, `fence_update`, and
  `fence_wait`.
- Keep ring/TCP smoke, JACCL smoke, and model TP proof as separate gates.

Current status: `PARTIAL`.

vMLX has Swift group/collective wrappers and a `TPRankWorker` smoke mode. The
current wrappers still contain native/fatal edges and need stronger diagnostics
before they should be considered production-grade cluster proof tooling.

### 3. Model TP Plan Discovery

Source references:

- `/Users/eric/adlab/scripts/engine-patches/adlab-qwen35-tp4-patch.py`
- `/Users/eric/adlab/scripts/engine-patches/adlab-qwen35-mlp-plan-patch.py`
- `/Users/eric/adlab/scripts/engine-patches/adlab-qwen35-ssm-exact-hit-patch.py`
- `/Users/eric/adlab/scripts/engine-patches/adlab-qwen3next-fused-qkv-patch.py`
- `/Users/eric/adlab/scripts/engine-patches/adlab-qwen3next-fused-gate-up-patch.py`
- `/Users/eric/adlab/scripts/engine-patches/adlab-mlx-lm-mimo-v2-flash-tp-shard-patch.py`
- `/Users/eric/adlab/scripts/engine-patches/adlab-sharded-vocab-greedy-patch.py`

Swift engine import targets:

- `Libraries/MLXDistributedTP/ShardingPlan.swift`
- `Libraries/MLXDistributedTP/ShardingPlans+Llama.swift`
- `Libraries/MLXDistributedTP/ShardingPlans+MiMoV2.swift`
- `Libraries/MLXDistributedTP/LinearLayers.swift`
- `Libraries/MLXDistributedTP/SwitchLinearLayers.swift`
- `tools/MiMoTPPlanProbe`
- future Qwen TP plan probe

Required Swift behavior:

- Qwen3.5/Qwen3.6 GatedDeltaNet TP plans must shard fused q/k/v ratios, depthwise
  conv, recurrent parameters, and MLP projections intentionally.
- MiMo V2.5 TP plans must preserve routed expert semantics, shared experts,
  attention sink bias, and rank-local source shard assumptions.
- Quantized affine/JANG/JANGTQ paths must not silently dequantize or reshuffle
  weights unless the model contract proves that is the configured path.
- Text-only TP proof must not silently drop VL/audio namespaces and claim full
  multimodal readiness.

Current status: `PARTIAL`.

The vMLX worktree already contains active MiMo TP plan files and a
`MiMoTPPlanProbe`. Qwen TP behavior is still mostly represented in AdLab Python
patch provenance and needs Swift-native source/test mapping before it can be
claimed as imported.

### 4. Live Proof Discovery

Source references:

- `/Users/eric/adlab/docs/mimo-v25-tp4-live-proof.md`
- `/Users/eric/adlab/docs/qwen36-tp4-current-status.md`
- `/Users/eric/adlab/docs/qwen36-mxfp8-tp4-noeval-speed-proof.md`
- `/Users/eric/adlab/scripts/adlab-swift-tp4-live-speed-gate.py`
- `/Users/eric/adlab/scripts/adlab-qwen36-tp4-resident-api-proof.sh`
- `/Users/eric/adlab/scripts/adlab-tp4-proof-verify.py`

Swift engine import targets:

- `tools/TPRankWorker`
- `tools/tp-launch.sh`
- `tools/tp-launch-2host.sh`
- future Swift live proof verifier or JSON schema tests

Required Swift behavior:

- Live proof must include worker liveness, API health, rank agreement, token
  authority, decode token/s, and real generated output.
- Cache proof must record prefix/L2/TurboQuant or companion cache evidence that
  matches the architecture.
- JACCL route proof must be separate from model correctness and speed proof.
- Multi-turn, streaming, parser/no-leak, and cache-hit gates belong above the
  raw worker layer and must not be inferred from a local collective smoke.

Current status: `TODO` for Swift-native verifier.

The live proof logic exists in AdLab Python and shell. vMLX needs a Swift-native
or package-owned verifier that can consume JSON artifacts from `TPRankWorker`
and the local API shim.

## Known AdLab Proofs To Preserve

### MiMo V2.5 TP4

Status: `PARTIAL` for vMLX import, `FIXED` in AdLab Python/Shell proof context.

Proven in AdLab:

- Swift `TPRankWorker` over JACCL/RDMA.
- Pod 1 TP4, ranks n1/n2/n3/n4.
- Thunderbolt loopback data plane with Tailscale only for control.
- `rank0_all_sum_token_broadcast_per_slot` authority.
- 4-rank agreement.
- Batch=1 decode around 39 tok/s.
- OpenAI-compatible chat, Responses, streaming, multi-turn, cache reuse, and L2
  disk restore in the AdLab API proof.

Still needs in vMLX:

- Swift package-owned proof gates.
- A durable Swift diagnostic for all-sum authority latency.
- Cache-store stability proof under the current Swift engine, not just AdLab
  shell/API artifacts.
- Clear separation between text-only TP, VL/audio tensor namespaces, and full
  multimodal MiMo readiness.

### Qwen3.5/Qwen3.6 TP4

Status: `PARTIAL`.

AdLab has Python and patch-script evidence for Qwen TP plans and speed/root
cause work. The Swift engine still needs a source-owned plan/probe/test mapping
for fused GatedDeltaNet sharding, exact SSM companion cache behavior, and any
MTP/MXFP/JANG-specific runtime branches.

## Immediate Todo

- [x] Add a Swift import/status ledger under `.agents/rdma-tb5/`.
- [x] Extend `DistributedProbe` with data-plane address classification:
      Thunderbolt loopback/link, local loopback, Tailscale, Wi-Fi/other, unknown.
- [x] Add source tests for the classification logic before touching runtime
      collective code.
- [ ] Add `MLX_IBV_DEVICES` matrix validation as a pure Swift parser/test first.
- [ ] Add a metallib fast-sync symbol check that does not false-fail from
      `grep -q`/SIGPIPE behavior.
- [ ] Add a Qwen TP plan import note mapping every AdLab Python patch marker to
      the expected Swift file and test.
- [ ] Add a live proof schema for `TPRankWorker` JSON output: rank, world size,
      backend, token authority, decode token/s, cache stats, and rank agreement.

## Blockers

- The local worktree is already heavily modified by other runtime work. Keep
  imports additive and inspect diffs before modifying shared files.
- Full TP4/JACCL proof requires the AdLab pod, not this single local Mac.
- Python `mlx_lm.sharded_load` behavior can guide Swift design, but it is not
  release proof for the Swift engine.
- Native MLX/JACCL failures can be process-fatal; crash prevention must happen
  before unsafe backend entry.
