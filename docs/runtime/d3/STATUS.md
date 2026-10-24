# D3 decision runtime integration

Scope: metadata-detected d3, d3-flash, d3-turbo; mixed affine 4/6/8-bit and dense JANGH2; Osaurus decision loading/API, never chat generation. Source reference: pinned vendor contract and parity fixtures in the private D3 runtime handoff (2026-10-22).

## Work and acceptance

- [ ] Validate complete load contract (tokenizer codes and tensor dtype remain live gates).
- [x] Swift metadata admission and readout SHA256: 9/9 real bundles; declared shape/mode/codes-count checked.
- [ ] Match tokenizer class, Unicode tokenization, prompt IDs and state split.
- [ ] Headless backbone; noncausal full attention; causal GDN; turbo prefix-bidirectional attention.
- [ ] Image/video preprocessing, timestamp groups and three-axis position parity.
- [ ] Choice/noul/score and turbo box/track response semantics; per-question errors.
- [ ] Quantization: module-declared affine widths; mixed-bit Hadamard paths.
- [ ] Osaurus discovery excludes decision bundles from chat, agents and speculation; dedicated request path.
- [ ] Focused regression checks; real bundle parity by model/quant/media type.
- [ ] Fresh Osaurus app proof, actual load/cancellation, repeated requests, memory and prompt throughput.
- [ ] PRs with exact source/bundle/artifact evidence; merge engine first then app pin.

No generation, output token/s, MTP, conversational KV reuse or chat tool behavior is claimed for a cache-free decision forward. Prompt throughput and peak/settled physical memory are required. StateOnce, if added, requires its own numeric parity; it must not silently reuse chat caches. No release tag is part of this work.

## Initial findings

Current engine base 419b03a7c has no decision runtime. Its Qwen35 LanguageModel constructs a vocabulary head for untied bundles and falls back to tied embedding logits otherwise. Neither behavior is valid for D3. Shared Qwen3VL video position indexing treats temporal groups as one uninterrupted block; timestamp-separated groups require separate grids.

## Work in progress

The first implementation adds strict decision metadata/readout admission, ordered JSON and exact option rendering, an explicit headless Qwen35 constructor, cache-free noncausal and prefix-bidirectional forwards, mixed JANGH preparation reuse, timestamped video groups and a pure integer Qwen3.5 position implementation. The timestamp-group path is opt-in for D3; existing generative video callers retain their existing processor contract pending separate regression qualification. Box geometry/marker readout is being wired. Builds are in progress; none of these rows is qualified yet.

## Executed first sweep (engine 0a17ee061)

- All six affine bundles executed seven typed/text/image/video requests each (42/42 completed; 42/42 same top choice as the same-weight reference on this small sample).
- All three JANGH2 bundles failed admission before weight allocation: their sidecars declare a headless decision runtime but deliberately omit the older `format`/`format_version` summary. The executable JANGH v2 header/plan is already in config.json. A narrow metadata-only admission path, retaining authoritative-plan conflict checks, is being added with negative tests.
- Video was not qualified: Swift encoded 3,171 tokens where the reference encoded 1,269. The decision lane missed the per-frame pixel cap and used the generic sampler. The correction uses vendor frame-index sampling and preserves already sampled frame timestamps.
- Media also exposed the existing CoreImage-vs-reference resize difference. Decision-only preprocessing now uses the existing uint8 antialiased bicubic implementation; generative callers are unchanged.
- Diagnostic `MLX_ENABLE_TF32=0` changed none of the five Flash-6M text probability vectors (max difference 0). This hypothesis was rejected; no process-wide precision override was added.
- Whole-repository Release tests hit pre-existing test build issues (`Testing` absent with CLT; `MLXArray.withEvalLockForTesting` absent in Release with Xcode). A focused package compiles the unchanged affected test sources against the actual production libraries. These failed builds are not counted as passing tests.

Corrected module build: `swift build -c release --target MLXVLM` passed. Focused Release XCTest package: 17 tests executed, zero failures, one optional generic-Qwen metadata fixture skipped in the first run. The package compiles the repository's unchanged D3DecisionTests and JANGHConfigurationPartitionTests against production MLXVLM/MLXLMCommon, avoiding unrelated test-target compilation failures. The same-weight runtime sweep is being rerun after these corrections; no corrected-family result is claimed from the compile/unit result alone.
