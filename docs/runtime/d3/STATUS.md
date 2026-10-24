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
