# vmlx-swift vs vmlx-swift-lm Deep Comparison - 2026-05-15

This compares the current `vmlx-swift` production-engine branch against the
live sibling checkout at `../vmlx-swift-lm`.

## Snapshot

- Current repo: `vmlx-swift`, branch `vmlx-0.31.3`, HEAD `da8830b` before this
  DSV4 template/audit patch set.
- Reference repo: `../vmlx-swift-lm`, branch `main`, HEAD `81c8ef7`, behind
  `origin/main` by 5 commits and dirty.
- Treat `../vmlx-swift-lm` as reference evidence, not clean upstream. It has
  useful fixes, but it also carries local experimental deltas and stale policy
  that must not be copied blindly.

The live sibling worktree also has in-flight cache-salt test edits and many
parallel local runtime file edits. Those edits are a moving signal, not a
merge source. Each behavior-affecting delta must either be ported with a focused
test and live model/cache proof, or explicitly rejected with the reason recorded
here.

## Package Shape

`vmlx-swift` is already the consolidated engine shape: vendored MLX, Jinja,
Transformers/tokenizers, MLXPress, distributed targets, RunBench, and focused
runtime tests live in one package.

`../vmlx-swift-lm` is much slimmer and still depends on external packages such
as `mlx-swift`, `Jinja`, and `swift-transformers`. A no-index package stat shows
`Package.swift` differs by `260 insertions / 495 deletions` when compared from
this repo to the sibling. That shape should not be imported wholesale; it would
undo the consolidated-engine goal.

Current-only runtime files found in the core library trees:

- `Libraries/MLXLMCommon/Cache/JangPressMachCache.swift`
- `Libraries/MLXLMCommon/Cache/MLXPressMmapColdSweep.swift`
- `Libraries/MLXLMCommon/TurboQuant/NumPyPCG64.swift`

These are part of the current engine surface, not things to remove just because
the slimmer sibling does not carry them.

## Ported Now

### ZAYA / ZAYA1-VL JANGTQ_K Metadata

The sibling had useful handling for nested `mxtq_bits.routed_expert` metadata.
This is a real correctness issue because JANGTQ_K can encode separate
`gate_proj`/`up_proj`/`down_proj` widths. If the runtime collapses that metadata
wrongly, it can dispatch the fused gate+up kernel with the wrong bit width or
lose the down-projection width.

Ported into `vmlx-swift`:

- `Libraries/MLXLLM/Models/Zaya.swift`
  - Directly decodes flat, per-role, and nested JANGTQ_K `mxtq_bits`.
  - Preserves `mxtq_gate_up_bits` and `mxtq_down_bits`.
  - Rejects mismatched `gate_proj` and `up_proj` instead of silently choosing
    one.
- `Libraries/MLXLLM/LLMModelFactory.swift`
  - ZAYA probe now uses `JSONDecoder.json5()`.
  - ZAYA probe reads nested projection bits and falls back to
    `config.textConfig.weightFormat` when the probe lacks `weight_format`.
  - `jang_config.json` merge now rejects mismatched gate/up widths instead of
    warning and continuing.
- `Libraries/MLXVLM/VLMModelFactory.swift`
  - VLM `jang_config.json` merge now handles nested `mxtq_bits`.
  - Top-level `mxtq_seed` is copied when not present in `quantization`.
  - Gate/up mismatch throws a configuration error.
- `Libraries/MLXVLM/Models/Zaya1VL.swift`
  - `mxtq_seed` is decoded, encoded, bridged into `ZayaTextConfiguration`, and
    used by the native ZAYA1-VL JANGTQ MoE context.

Active test coverage added:

- `Tests/MLXLMCommonFocusedTests/ZayaConfigDecodeFocusedTests.swift`
- `Tests/MLXLMCommonFocusedTests/VLShapeGuardFocusedTests.swift`

Legacy test coverage also updated in `Tests/MLXLMTests/ZayaConfigDecodeTests.swift`,
but that older test folder is not wired into the current package manifest. The
active coverage is the focused target.

Verification:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
swift test --filter 'ZayaConfigDecodeFocusedTests|VLShapeGuardFocusedTests|DeepseekV4ReasoningPolicyTests' --jobs 2
```

Result: 16 tests in 3 suites passed.

### DSV4 Standalone Jinja Tool Rendering

The sibling `origin/main` had a real DSV4 fallback-template fix for tool schemas
when no system message is present. `vmlx-swift` already had the corrected Swift
fallback string in `ChatTemplateFallbacks.dsv4Minimal`, but the repo-visible
standalone file `Libraries/MLXLMCommon/ChatTemplates/DSV4Minimal.jinja` was
behind it. That is a bad release surface because downstream maintainers and bundle
writers can inspect or copy the standalone file.

Ported into `vmlx-swift`:

- `Libraries/MLXLMCommon/ChatTemplates/DSV4Minimal.jinja`
  - Uses the same tool-rendering macro shape as the Swift fallback.
  - Renders top-level tools before the first user turn when no system message is
    present.
  - Keeps no-hidden-downgrade reasoning behavior: `enable_thinking=true` opens
    `<think>`, direct chat mode emits `</think>`, and `reasoning_effort=max`
    remains visible to the template.
- `Tests/MLXLMCommonFocusedTests/DeepseekV4ChatTemplateFallbackFocusedTests.swift`
  - Reads the standalone `.jinja` file and proves no-system OpenAI tools render
    into a DSML schema block before the user turn.

Verification:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
swift test --filter DeepseekV4ChatTemplateFallbackFocusedTests --jobs 2
```

Result: 6 tests in 1 suite passed. A plain `swift test` invocation without
`DEVELOPER_DIR` still fails on this machine because the default toolchain cannot
import Swift Testing.

## Already Aligned

These files have no current no-index source diff against the sibling snapshot:

- `Libraries/MLXLMCommon/BatchEngine/BatchEngine.swift`
- `Libraries/MLXLMCommon/BatchEngine/BatchScheduler.swift`
- `Libraries/MLXLMCommon/Cache/CacheHelpers.swift`
- `Libraries/MLXLMCommon/Cache/MediaSalt.swift`
- `Libraries/MLXLMCommon/TurboQuantSwitchLinear.swift`

That does not prove production behavior, but it means there is no obvious source
delta to port for those files in this comparison pass.

## File Delta Ledger

This is the current no-index source comparison status for the core files that
still differ from `../vmlx-swift-lm`.

| File or area | Delta | Action |
| --- | --- | --- |
| `Libraries/MLXLLM/LLMModelFactory.swift` | Small residual diff after ZAYA port. Remaining difference is mostly comment/policy around Kimi/DSV3 MLA. | Keep current until real Kimi/DSV3 decode proof says the fp32 L==1 helper is unnecessary. |
| `Libraries/MLXLLM/Models/DeepseekV3.swift` | Current uses `mlaScaledDotProductAttention`; sibling uses `attentionWithCacheUpdate`. Current also has stricter grouped-router masking. | Do not port blindly. This affects Kimi/DSV3 decode numerics and cache update semantics. Needs side-by-side model smoke. |
| `Libraries/MLXLLM/Models/DeepseekV3JANGTQ.swift` | Current carries MLXPress streaming/profiling, per-projection gate/down bits, chunked MoE, and cold-sweep hooks; sibling is much simpler. | Keep current for now. Validate by MiniMax/Kimi/Qwen JANGTQ active expert rows, not source preference. |
| `Libraries/MLXLLM/Models/DeepseekV4Compressor.swift` and `DeepseekV4MathHelpers.swift` | Sibling is much smaller. Current carries more DSV4 helper code. | Needs DSV4 long-context/vector drift review before any deletion or port. |
| `Libraries/MLXLLM/Models/Hy3.swift` | Current has `MLXPressMmapColdSweep.afterLayer`; sibling removes it. | Keep current. This is part of low-footprint cleanup; only remove if a live Hy3 row proves it hurts correctness or speed. |
| `Libraries/MLXLLM/Models/MiniMaxJANGTQ.swift` | Current carries more JANGTQ_K/per-role handling than sibling. | Keep current pending MiniMax active routed expert validation. |
| `Libraries/MLXLMCommon/ModelFactory.swift` | Current makes permanent prestack explicit opt-in and keeps tensor-buffer mmap diagnostic gated; sibling runs prestacker more broadly and enables tensor mmap when JangPress mmap is on. | Keep current. Sibling behavior risks hidden full-footprint or stale prestack dependence; needs low-RAM proof before changing. |
| `Libraries/MLXLMCommon/Load.swift` | Current treats `weight_format=mxtq` / JANGTQ profile as native even without sidecar because `JANGTQRuntimeCache` can generate deterministic signs/codebooks. Sibling requires sidecar and fail-fast errors. | Open. This is policy-sensitive: deterministic generation is real code, not a sampling guard, but missing-sidecar bundles still need explicit runtime proof. |
| `Libraries/MLXLMCommon/JANGTQKernels.swift` | Large current-only implementation around PCG64/codebook and newer kernels. | Keep current. Test by kernel-focused unit tests plus live JANGTQ decode rows. |
| `Libraries/MLXLMCommon/JANGTQStreamingExperts.swift` | Very large current-only active-streaming implementation. | Keep current while auditing missing tensor diagnostics, active bank selection, offset-addressed kernels, and low-footprint behavior. |
| `Libraries/MLXLMCommon/JangLoader.swift` | Current has more loader/template-sidecar behavior. | Review with VL template and JANG bundle config matrix. |
| `Libraries/MLXLMCommon/Evaluate.swift` | Current is larger and carries current profiling/cache/sampling/reasoning work. | Review route by route; do not port sibling simplification until API behavior passes. |
| `Libraries/MLXLMCommon/ChatTemplates/ChatTemplateFallbacks.swift` | Current is much larger. | Keep current; verify with DSV4, MiniMax, ZAYA-VL, tool-calling, and reasoning matrix. |
| `Libraries/MLXLMCommon/DeepseekV4ChatEncoder.swift` | Sibling comments assume public max is normalized to high. Current preserves raw pass-through. | Keep current per no-hidden-downgrade policy. |
| `Libraries/MLXLMCommon/DeepseekV4ReasoningPolicy.swift` | Sibling downgrades low/medium/max into high by default. | Rejected. Current focused tests preserve pass-through. |
| `Libraries/MLXVLM/Models/NemotronHOmni/NemotronHOmni.swift` | Only arrow/comment punctuation differs. | No runtime action. |
| `Libraries/MLXVLM/Models/Zaya1VL.swift` | Useful seed/nested-bit pieces were ported while preserving this repo's stronger dynamic decoder. | Done for metadata. Still needs live VL multi-turn proof. |

## Explicitly Rejected From vmlx-swift-lm

### DSV4 Reasoning Downgrade Policy

The sibling carries a `DeepseekV4ReasoningPolicy` shape that can alias public
`reasoning_effort=max`/`low`/`medium` into another value unless gated by an
environment variable. That is a hidden behavioral guard and conflicts with the
current production rule: if a caller asks for `max`, it must reach the encoder as
`max`.

Current `vmlx-swift` keeps the pass-through policy. The focused tests prove:

- public `max` passes through without downgrade;
- `low`, `medium`, and `high` remain distinct;
- direct/no-think rails remove reasoning effort intentionally;
- non-DSV4 contexts are unchanged.

### Sibling Package Slimming

The sibling package graph is not a target state for this repo. `vmlx-swift` must
replace and consolidate MLX/Jinja/Transformers-style dependencies for Osaurus,
not move back to external repo stitching.

### Historical Speed Claims

Old JangPress and MiniMax speed rows remain useful as targets, not proof. A row
is not production-ready unless it has current low physical footprint, tok/s,
coherent visible output, no loop/length-cap fake pass, multi-turn behavior, and
cache-topology proof.

## High-Risk Deltas Still Requiring Deep Review

These are not ignored. They are the next comparison lanes because they touch
runtime behavior and cannot be safely copied by stat/diff alone.

| Area | Current comparison state | Required proof before accepting |
| --- | --- | --- |
| `JANGTQStreamingExperts.swift` | Huge source delta; this repo has the larger implementation. | Sidecar missing-array behavior, active-streaming low-RAM path, resident vs ephemeral overlays, no full-footprint hidden load, token/s and multi-turn coherence. |
| `Evaluate.swift` | Large delta; this repo carries current MLXPress profiling, KV-policy salting, and DSV4 reasoning pass-through work. | HTTP/BatchEngine/TokenIterator parity, sampling defaults, reasoning on/off, tool stream splitting, no hidden clamps. |
| `DeepseekV4ChatEncoder.swift` and `DSV4Minimal.jinja` | Standalone no-system tool rendering is fixed here; encoder and long-context behavior remain policy-sensitive. | `enable_thinking`, `reasoning_effort=max`, CSA/HSA/SWA long context, prefix/paged/disk cache proof, vector drift check. |
| `BatchEngine.swift` / `BatchScheduler.swift` | No current no-index source diff against the sibling snapshot. | Still requires live B>1 continuous batching, cancel, mid-stream failure, cache isolation by policy salt, and hybrid cache behavior. |
| `MediaSalt.swift` and VL path | `MediaSalt.swift` is source-aligned; VLM model/factory files still differ. | Image+text -> text-only -> new image multi-turn, nil media salt on text-only turn, prefix resume correctness, grounded output. |
| `Hy3.swift` | Sibling dirty delta exists. | Real Hy3 local smoke with reasoning on/off, correct chat template, no repeated empty think blocks, cache stack behavior. |
| `JangPress*` cache files | Sibling dirty deltas exist; this repo also has newer Mach/cache cold-sweep files. | Prove active routed experts, low physical footprint, useful decode speed, no per-token SSD bottleneck, coherent multi-turn output. |

## Production Readiness Implication

This comparison pass does not make the whole engine production-ready. It closes
one concrete correctness gap: ZAYA/ZAYA1-VL nested JANGTQ_K metadata and seed
plumbing now match the useful sibling behavior while preserving this repo's
stricter no-fake-guard policy.

The remaining production gate is behavioral, not just source parity:

- each model family needs live multi-turn generation;
- each attention/cache topology needs prefix/paged/L2/SSM/VL proof where
  applicable;
- DSV4 needs long-context CSA/HSA/SWA, prefix/paged/disk, pool/finalizer, and
  vector drift proof;
- MiniMax/JANGTQ needs low-footprint active expert behavior plus coherent
  multi-turn speed rows;
- VL/video/audio paths need real media payloads and grounded multi-turn cache
  validation;
- API and Osaurus wiring must be tested through the live routes, not just by
  reading source.

## Next Comparison Order

1. `JANGTQStreamingExperts.swift` and JangPress cache files: identify whether
   sibling deltas improve real active-routed low-RAM behavior or are stale.
2. `Evaluate.swift`, `BatchEngine.swift`, `BatchScheduler.swift`: isolate
   behavior-affecting scheduler/cache/sampling changes.
3. `DeepseekV4ChatEncoder.swift`, `DSV4Minimal.jinja`, DSV4 tests: prove no
   hidden reasoning downgrade and no long-context cache regression.
4. `MediaSalt.swift`, `Zaya1VL.swift`, VLM factory/tests: complete the VL
   multi-turn media-cache proof.
5. Osaurus staging wiring: compare pins and route expectations after the engine
   source lanes are resolved.
