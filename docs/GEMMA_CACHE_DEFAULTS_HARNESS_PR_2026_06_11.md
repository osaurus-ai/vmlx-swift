# Gemma Cache Defaults and Harness PR Status

Date: 2026-06-11

This branch is focused only on the Gemma cache-default and Osaurus harness
compatibility lane.

## Goals

1. Turn paged RAM KV cache off by default for normal model loads. Paged cache is
   useful for explicit multi-batch/cache-regression lanes, but it does not help
   most single-batch user generations enough to justify default RAM footprint.
2. Keep TurboQuant KV cache on by default for Gemma JANG and MXFP bundles from
   both Osaurus chat and Osaurus server-settings paths. The proof must come from
   vMLX runtime settings and cache telemetry, not output coercion.
3. Compare unquantized Gemma BF16/source bundles against QAT/MXFP/JANG
   quantized Gemma bundles in the Osaurus harness compatibility evals.
4. Add runtime prefill progress events so Osaurus can show prompt-processing
   percentage before first token, instead of leaving users staring at a blind
   wait.

## Current Understanding

- The source default for paged RAM cache is `VMLXPagedKVCacheSettings(enabled:)`
  in `Libraries/MLXLMCommon/ServerRuntimeSettings.swift`.
- The lower-level direct runtime default is `CacheCoordinatorConfig(usePagedCache:)`
  in `Libraries/MLXLMCommon/Cache/CacheCoordinatorConfig.swift`.
- Memory safety previously re-enabled paged RAM KV whenever prefix cache was
  enabled. That must remain off unless the caller explicitly opts into paged KV.
- The Osaurus-facing server settings bridge resolves `.engineSelected` live KV
  codec into `CacheCoordinatorConfig.defaultKVMode == .turboQuant(...)`.
- Existing cache topology tests intentionally force `usePagedCache: true` for
  paged-regression coverage. Those tests should remain explicit and should not
  define the production default.
- Osaurus harness compatibility requires real agent-loop eval rows with
  pass/fail counts and causes attached. Harness bugs must be fixed before
  publishing a model row; model failures should be recorded honestly.
- Prefill currently happens before the first generated token in both
  `TokenIterator` and `BatchEngine`. The existing generation streams do not
  expose prefill progress events, so the UI cannot distinguish a slow prompt
  prefill from a frozen model.

## Planned Proof

- Focused source tests:
  - Default `VMLXServerRuntimeSettings` has `cache.pagedKV.enabled == false`.
  - Default `cacheCoordinatorConfig()` has `usePagedCache == false`.
  - Default settings still resolve `defaultKVMode` to TurboQuant KV.
  - Explicit paged-cache opt-in still works for multi-batch/regression callers.
- Runtime/bench proof:
  - Gemma load/generation row records token/s and cache topology.
  - Gemma JANG/MXFP rows show TurboQuant KV layers in topology/telemetry.
  - No row is called fixed without visible coherent output, token/s, and the
    relevant cache evidence.
- Osaurus harness proof:
  - Run AgentLoopFrontier and AgentLoop for each Gemma route chosen for the PR.
  - Compare BF16/source rows against quantized Gemma rows.
  - Add/update Osaurus docs only after the eval output is real and attributed.
- Prefill progress proof:
  - Source tests verify progress events are ordered, monotonic, and bounded
    from 0...1 for token-only, cache-hit, and media paths.
  - Live Osaurus chat/API rows show prefill progress before first token on a
    long prompt, then normal token streaming with token/s in final info.
- Final app-facing merge gate:
  - Build the unsigned/dev Osaurus app without keychain or signing prompts.
  - Load at least one Gemma 4 QAT MXFP4/JANG_4M model from `~/models`.
  - Chat with it in Osaurus and verify visible coherent multi-turn output.
  - Exercise a real Osaurus tool call and verify exact tool name/arguments and
    tool-result continuation inside the app.
  - Record prefill progress visibility, token/s, cache topology, and RAM /
    physical-footprint observations during load and generation.

## Status

- Worktree: `/Users/eric/.config/superpowers/worktrees/vmlx-swift/cache-defaults-bf16-qat`
- Branch: `codex/cache-defaults-bf16-qat`
- Base: `vmlx-origin/main` at `76047f3b`
- Implementation: not yet complete.
- Implementation status:
  - Paged RAM KV default flipped off in `VMLXPagedKVCacheSettings`.
  - Direct `CacheCoordinatorConfig()` default flipped off.
  - Public `MLXPressCacheConfiguration` default flipped off for paged RAM KV.
  - Memory-safety plan no longer silently turns paged RAM KV back on when
    prefix cache is enabled.
  - Engine-selected live KV remains TurboQuant by default.
  - Gemma SWA cache contract is explicit: full-attention `KVCacheSimple` layers
    are TurboQuant-eligible; sliding/SWA layers stay `RotatingKVCache` and use
    disk-backed restore.
  - Batch generation now emits `PrefillProgress` stage events before first
    decoded token.
- Verification:
  - `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift build --target MLXLMCommon -Xswiftc -F -Xswiftc /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks` passes.
  - `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test --filter coordinatorDefaultsKeepPagedRAMCacheOff --jobs 2 -Xswiftc -F -Xswiftc /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks` passes.
  - `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test --filter defaultsPreserveEngineAndBundleSamplingDecisions --jobs 2 -Xswiftc -F -Xswiftc /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks` passes.
  - `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test --filter automaticRuntimeCachePolicyCoversDownloadedArchitectureFamilies --jobs 2 -Xswiftc -F -Xswiftc /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks` passes.
  - `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test --filter explicitPagedCacheOptInStillEnablesCoordinatorPagedTier --jobs 2 -Xswiftc -F -Xswiftc /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks` passes.
  - `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test --filter memorySafetyPreservesHybridSSMAndEngineSelectedCacheTopology --jobs 2 -Xswiftc -F -Xswiftc /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks` passes.
  - `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test --filter BatchQuantizeHookTests/testGemmaSWATopologyContract --jobs 2 -Xswiftc -F -Xswiftc /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks` passes.
  - Broad suite filter still times out under the Swift Testing helper; use
    function-name filters for focused proof until that harness issue is
    resolved.
  - `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift run mlxpress-selfcheck ...` builds, but runtime is blocked by MLX metallib discovery:
    `MLX error: Failed to load the default metallib. library not found`.
- Prefill progress spec: `docs/PREFILL_PROGRESS_RUNTIME_CONTRACT_2026_06_11.md`

## Local Model Download Queue

Requested target: `~/models`.

- Downloader helper: `scripts/download_gemma4_qat_models.py`
- Detached runner: completed `screen` session `gemma4_qat_download`
- Active log dir:
  `/Users/eric/models/.download-logs/gemma4-qat-screen-20260611T222059Z`
- Current proof:
  - All 10 requested OsaurusAI Gemma 4 QAT repos completed.
  - Sizes:
    - `OsaurusAI--gemma-4-12B-it-qat-JANG_4M`: 9.5G
    - `OsaurusAI--gemma-4-31B-it-qat-JANG_4M`: 25G
    - `OsaurusAI--gemma-4-26B-A4B-it-qat-JANG_4M`: 17G
    - `OsaurusAI--gemma-4-E4B-it-qat-JANG_4M`: 10G
    - `OsaurusAI--gemma-4-E2B-it-qat-JANG_4M`: 7.3G
    - `OsaurusAI--gemma-4-31B-it-qat-MXFP4`: 18G
    - `OsaurusAI--gemma-4-26B-A4B-it-qat-MXFP4`: 15G
    - `OsaurusAI--gemma-4-12B-it-qat-MXFP4`: 7.4G
    - `OsaurusAI--gemma-4-E4B-it-qat-MXFP4`: 5.5G
    - `OsaurusAI--gemma-4-E2B-it-qat-MXFP4`: 3.8G

## Boundaries

- Do not expand into unrelated model families while this lane is open.
- Do not add forced behavior fixes, hidden sampler overrides, prompt coercion,
  or fake parser cleanup to make Gemma outputs look better.
- Do not claim TurboQuant KV unless the runtime settings and cache telemetry
  prove it.
- Do not show fake prefill percentages from wall-clock timers. If a model path
  cannot provide fine-grained progress yet, emit truthful stage-level progress
  and document the limitation.

## Open Items

- Find/download the Gemma 4 BF16/source models for the harness comparison rows.
  Current `~/models` inventory has the 10 QAT MXFP4/JANG_4M repos but no
  obvious BF16/source Gemma 4 directories.
- Commit/push or otherwise publish the vMLX change so Osaurus can pin a SHA
  containing `Generation.prefillProgress`.
- Re-run Osaurus focused tests after repinning to the updated vMLX SHA.
- Run the final Osaurus app-facing gate: dev app build, model load, chat,
  tool-call proof, token/s, cache topology, prefill progress, and RAM footprint.
