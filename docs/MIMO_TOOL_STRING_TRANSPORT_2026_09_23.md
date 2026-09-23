# MiMo tool string transport

MiMo's installed template renders string arguments directly between parameter tags. The SGLang MiMo parser likewise preserves boundary newlines and literal backslashes. The shared Qwen XML parser instead stripped boundary CR/LF and JSON-unescaped string values. In the Osaurus AgentLoopFrontier byte-exact-write case, dispatched writes lost trailing newlines and newline-only append attempts became empty strings.

The correction gives MiMo a distinct `mimo` tool format using literal string transport. MiMo architecture inference and capability aliases select it; older `xml_function` stamps are refined when the declared model type is MiMo. Both language and vision factories pass that architecture when resolving newer chat-level stamps. Explicit other formats and non-MiMo XML behavior stay unchanged. No model, quantization, sampler, template or generated answer is modified.

## Evidence

Private evidence root: `~/vmlx-private-evidence/mimo26-swift-2026-09-22/`.

- `r27-local-baseline-summary.json`: app source `51358261c284880dfc9203a0b6ac2a6c3ae78419`, runtime pin `fce53ef0e5cf5eb052a5a38490661bc48218917f`; ReasoningChannel 13/13, CacheProof 14/14, interrupted AgentLoopFrontier 8 passed / 1 failed. Stopped deliberately after the parser defect was identified, not a complete matrix.
- `native-tool-values-r28-red-r2-receipt.json` and log: production Swift parser/streaming regression fails with nine assertions, including newline-only strings becoming empty, trailing newline loss and literal backslash-n becoming a newline. Initial invocation's missing-scheme error is retained separately.
- `native-tool-values-r28-green-receipt.json`: the first correction passes five tests, including a five-value parameterized case, streaming delivery, legacy stamp routing and Qwen/JSON compatibility.
- `r28-upstream-string-reference.json`: upstream MiMo conversion function preserves the five string fixtures. Source blob `1af745c82d887111b532aafd82b4df111e2219f8` from [SGLang MiMo detector](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/mimo_detector.py). This bounded function check is not a full SGLang model run.

On current main `27c50c4915c89a569823305f7f82f08add9be48b`, the expanded run completed 108 tests across five suites with four pre-existing M5 TF32 known issues. The required separate `MLX_ENABLE_TF32=0` process passed all 108 tests with no known issues. Both exited 0 without guard trips. Receipts: `native-tool-values-r28-main-regression-receipt.json` and `native-tool-values-r28-main-strict-regression-receipt.json`. The strict setting applies only to this numerical regression process; production model defaults are unchanged. Osaurus must consume the merged fix, rebuild, and rerun affected full-model and development-app proof; a unit test alone does not close app qualification. Audio quality failures remain separately documented. No release or tag is authorized.
