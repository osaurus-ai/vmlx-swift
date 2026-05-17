# Osaurus speculative-decoding integration

Block-diffusion speculative decoding (DFlash + DDTree) for vmlx-swift-lm.
Osaurus enables it by setting one field on `GenerateParameters` — no other
API changes.

## 1. The one-line integration

```swift
var params = GenerateParameters(maxTokens: 256, temperature: 0)
params.draftStrategy = .ddtree(
    drafterPath: URL(fileURLWithPath: "…/z-lab/Qwen3.5-27B-DFlash"),
    branchingBudget: 32,
    blockSize: 16)

// Works with BOTH Evaluate.generate and BatchEngine.generate:
let stream = try generate(input: input, parameters: params, context: ctx)
// or
let stream = await engine.generate(input: input, parameters: params)

for await event in stream {
    switch event {
    case .chunk(let text):     // pure user text, reasoning-stripped, tool-stripped
    case .reasoning(let r):    // streaming <think>…</think> delta
    case .toolCall(let call):  // fully parsed tool call
    case .info(let info):      // prompt_tokens + generation_tokens + wall time
    }
}
```

When `draftStrategy = nil` or `.none`, every existing code path runs byte-identically to today. Opt-in only.

## 2. DraftStrategy enum

```swift
public enum DraftStrategy: Sendable {
    /// No spec-dec. Byte-identical to upstream.
    case none

    /// Classic autoregressive draft-model (existing SpeculativeTokenIterator path).
    case autoregressive(draftModel: any LanguageModel, numDraftTokens: Int)

    /// DFlash block-diffusion drafter + linear verify.
    case dflash(drafterPath: URL, blockSize: Int)

    /// DDTree — DFlash drafter + best-first heap tree verify.
    /// Strict superset of .dflash; the tree with branching=1 reduces
    /// to DFlash linear.
    case ddtree(drafterPath: URL, branchingBudget: Int, blockSize: Int)
}
```

- `blockSize` must match the drafter's training `block_size` (read from its `config.json`).
- `branchingBudget` is the max tree nodes (excluding root); paper recommends 32-64 for greedy, 16-24 for sampling.

## 3. Byte-parity invariant

At temperature 0, the committed token sequence from `.dflash` / `.ddtree` is **byte-identical** to plain greedy autoregressive decode. The drafter affects SPEED (mean acceptance length), not OUTPUT.

This invariant is pinned by:
- `DFlashLinearByteParityTests` (Phase 1 iter 5)
- `DDTreeEndToEndTests` (Phase 2 iter 9)

If osaurus observes drift vs `.none` output at temp 0 on the same seed, that's a bug — file an issue.

## 4. Drafter checkpoints (z-lab on Hugging Face)

| Repo | Target | Drafter size | Status |
|---|---|---|---|
| `z-lab/gpt-oss-20b-DFlash` | gpt-oss-20b (dense) | 1.5 GB | public |
| `z-lab/Qwen3.5-27B-DFlash` | Qwen 3.5-27B (hybrid SSM) | 3.2 GB | public |
| `z-lab/Kimi-K2.5-DFlash` | Kimi K2.5 | large | public |
| `z-lab/Qwen3-8B-DFlash` | Qwen 3-8B | — | 401 gated / unreleased |
| `z-lab/Llama-3.1-8B-Instruct-DFlash` | Llama 3.1 8B | — | 401 gated / unreleased |

Download with `hf download z-lab/<name> --local-dir <dir>`.

## 5. Target model requirements

For `.dflash` / `.ddtree` to activate, the target model must conform to both:

- `HiddenStateCaptureModel` — exposes per-block hidden states the drafter reads via `fc` + KV injection.
- `TokenEmbedderModel` — exposes `embed(_:)` + `projectToLogits(_:)` the drafter shares for its noise input and draft-logit projection.

Current vmlx conformance:

| Model family | HiddenStateCaptureModel | TokenEmbedderModel |
|---|---|---|
| Qwen3 | ✅ | ✅ |
| Qwen 3.5 (hybrid SSM) | ⏳ Phase 3 | ⏳ Phase 3 |
| Qwen 3.6 JANGTQ | ⏳ Phase 3 | ⏳ Phase 3 |
| Gemma 4 / E2B / E4B | ⏳ | ⏳ |
| GPT-OSS 20B | ⏳ | ⏳ |
| Llama 3.x | ⏳ | ⏳ |

Non-conforming targets fall through to the non-speculative path — `Evaluate.generate` / `BatchEngine.generate` return the plain stream with `draftStrategy` silently ignored. No error, no regression.

## 6. Drafter resolver

`SpecDecDrafterResolver` caches loaded drafters by disk path. `SpecDecDrafterResolver.shared` is a package-level singleton the dispatch uses automatically. Osaurus can provision its own instance for a bounded cache lifetime:

```swift
// Per-container resolver (cleared on model unload).
let resolver = SpecDecDrafterResolver()
// Custom dispatch via SpecDecStream.streamViaStrategy(resolver:…) is public.
```

Cache eviction: `resolver.evict(path:)` drops one; `resolver.evictAll()` drops everything.

## 7. What lives at the library level

Same contract as tool-call parsing: everything happens inside vmlx-swift-lm. Osaurus does not re-parse, does not run its own drafter loop, does not read drafter safetensors. The stream it consumes is the same `AsyncStream<Generation>` it already consumes.

## 8. Streaming contract

`AsyncStream<Generation>` yields the same events as the non-speculative path:

- `.chunk(String)` — user-visible text. Reasoning (`<think>...</think>`) peeled off into `.reasoning` events; tool-call envelopes extracted into `.toolCall` events.
- `.reasoning(String)` — streaming chain-of-thought delta, one per parser segment. Concatenate consecutive `.reasoning` events for the full think block.
- `.toolCall(ToolCall)` — authoritative tool call extracted by the library.
- `.info(GenerateCompletionInfo)` — one at end. Reports prompt token count + generation token count + wall time.

## 9. Performance expectations

| Scenario | DFlash (expected) | DDTree (expected) |
|---|---|---|
| Dense target, short block | 3× over AR | 4-5× over AR |
| Dense target, long block | 4-6× over AR | 5-7× over AR |
| Hybrid SSM target | 1.3× over AR | 1.5× over AR (Phase 3 removes the SSM ceiling) |

These are paper-reported numbers on CUDA. Swift/MLX on M-series will differ — real-checkpoint benchmarks land in Phase 4 iter 14+ and get pinned in `DDTREE-DESIGN.md` with commit SHAs.

**Current state**: iter 10-11 ships the integration path but Phase 2 iter 8's tree-verify is v1 multi-run (correct but slow — same speed as AR). Single-forward tree-verify with combined `(1, 1, T, prefix_len + T)` attention mask is iter 13+ optimization work.

## 10. JANG capability stamp (planned, Phase 5)

JANG converters will emit:

```json
{
  "capabilities": {
    "draft_strategy": "ddtree",
    "drafter_path": "drafter/",   // relative to jang_config.json
    "branching_budget": 32
  }
}
```

`ParserResolution.draftStrategy(capabilities:)` will auto-populate `ModelConfiguration.draftStrategy` so osaurus flips the feature on by loading a JANG-stamped model — no per-request config needed.

## 11. BatchEngine integration

`BatchEngine.generate(input:parameters:)` supports `draftStrategy` at the API
surface:

- `.dflash` / `.ddtree` dispatch through `SpecDecStream.streamViaStrategy`
  when the target and drafter meet the block-diffusion contracts.
- `.nativeMTP(depth:)` dispatches through the native MTP iterator as an
  exclusive solo lane. This prevents a false AR fallback when Osaurus requests
  MTP, but it is not a multi-slot paged native-MTP scheduler.
- `BatchEngine.submit(input:parameters:)` rejects native MTP. Raw batched
  native-MTP scheduling remains future work.

Per-slot strategies in the batched path are Phase 4 late work.

## 12. Verification

- Unit tests: `swift test --filter 'DDTree|DFlash|SpecDec'` — 65 green as of iter 11.
- Byte-parity: `DFlashLinearByteParityTests`, `DDTreeEndToEndTests`.
- Real-checkpoint load: `DFlashDrafterForwardTests.testLoadGptOssDrafter` / `.testLoadQwen35Drafter` (skipped when drafter not on disk).
- Full sweep: `./scripts/verify-engine.sh --tests-only` — 121 baseline + SpecDec all green.

## 13. What's NOT yet

| Gap | Resolution plan |
|---|---|
| Hybrid SSM (Qwen 3.5 / 3.6, Nemotron-H) target conformance | Phase 3 iter 13-14 |
| Single-forward tree-verify (vs iter-8 multi-run) | Phase 2 iter 13 |
| Per-slot strategies in batched decode | Phase 4 iter 15-18 |
| JANG capability-stamp auto-pickup | Phase 5 iter 19-20 |
| Real-checkpoint tok/s measurements in DDTREE-DESIGN.md | Phase 4 iter 14+ |

See `DDTREE-DESIGN.md` for the detailed iter log with commit SHAs.
