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

## Expanded parity findings (ae5ab4bb95)

All nine bundles completed the seven-row smoke check: exact prompt IDs and image/video grids, 63/63 identical top decisions on those rows. This is not complete qualification. The expanded Flash-4M suite found a substantial image probability/decision difference and stopped at a decomposed-Unicode long document because boundary lookup used raw bytes instead of the tokenizer's declared NFC normalization. The latter is corrected in source and has a focused regression test; the image tensor comparison is still under investigation. The stale-source full runner was stopped after this common failure, preserving completed rows rather than repeatedly running the known failure on every quant.

The new uint8 image path initially returned HWC instead of the NCHW consumed by patchification. All nine failed image rows from f49a4d2d2 are preserved; ae5ab4bb95 corrects the frame axis/channel order. A nonsquare known-colour regression now checks actual processor patch layout and channels.

Initial endpoint scope is choice/noul/score, state-level images/videos and Turbo image boxes. Extended vendor asset composition/video-at/track and StateOnce are explicit follow-ups; unsupported endpoint fields must fail rather than be silently ignored. This does not qualify those extensions.

## Corrections and current qualification run

`5fdc30d34c` selects D3's declared GELU-tanh vision MLP and bounds attention by exact 2048-query chunks. The failed Flash-4M image decision now agrees with the independently corrected Python reference (Swift p=0.63524, Python p=0.64040); the NFC long document completes. Turbo image-box and long-text/media targeted rows complete. The full nine-bundle fixture suite is running from this revision. Flash-4M completes 195/195 with no non-near-tie decision mismatches, one near-tie flip, median max probability difference 0.00227. Do not interpret incomplete rows from the other bundles as a pass.

The focused suite now executes 19 XCTest cases with zero failures, including actual known-colour nonsquare patchification and NFC token-boundary lookup. One optional metadata fixture was skipped in this run; it passed separately earlier. A missing test-runner metallib and two source-modified-during-build failures are retained as failed infrastructure/build attempts, not counted as passing tests.

This branch preserves generic generative defaults. Their shared timestamped-video position path and declared vision activation still need a separate regression lane; the D3 correction is opt-in. API scope/extensions and the bundle minimum-version nuance must be disclosed in the app PR. No current app, memory-safety, cancellation or merge-ready claim follows from diagnostic-only results.
