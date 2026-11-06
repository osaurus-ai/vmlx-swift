# DFlash request fallback and GLM image-token validation

A GLM image request with Adaptive enabled and batch size two could finish cancelled without decoding. The target has no DFlash media-prefill implementation, so the request is ineligible for drafting. `BatchEngine.generate` skipped the exclusive draft path but retained its strategy; the raw batch submission then correctly rejected an unsupported draft cycle.

Clear the ineligible strategy before ordinary submission. Media preparation and every generation parameter remain intact. Eligible exclusive DFlash decoding, the existing busy-engine AR fallback and direct raw-submit rejection remain unchanged. No model-family exception, sampler override, prompt coercion or speculative-media capability is invented.

Separately, the GLM processor accepted any non-nil required-image vocabulary lookup. Use the existing media-token resolver to require a single token that decodes to the exact image placeholder. Explicit-ID construction and optional delimiter/video lookup are unchanged. The baseline accepts three invalid fixtures: unknown alias, split literal and wrong decode. The candidate passes all constructor controls and five actual local tokenizer/history/tool-prefix/media-marker checks; canonical sidecars remain unchanged. An initial tokenizer suite was skipped by Xcode environment forwarding and is not counted.

The source-bound Release41 baseline preserves three real image requests cancelled with zero output, including a submit immediately after a completed text request with resident weights confirmed. Two text controls answered correctly. These are failed media rows, not image qualification.

The new tiny-model comparison covers media, presence/frequency/repetition penalties, suppression, min-p, one-token output and unusable draft width at batch sizes one and two. All eight batch-two baseline variants cancel empty; their ordinary references and batch-one controls decode. All16 candidate combinations match explicit AR, including pixel-dependent preparation. The final formatted fixture passes the same16 cases. This is dispatch/parameter evidence, not a speed or real-model quality benchmark.

The full normal Xcode focused run reports1257functions:1213pass,7known failed methods and37skip. There are no new failed methods; the two registry failures were resolved separately by #740. The full target remains red. No existing assertions or opt-in gates are weakened.

Current source-bound app qualification and landing receipts are tracked in [PR741](https://github.com/osaurus-ai/vmlx-swift/pull/741). Constructor and component passes alone do not establish GLM media quality, cache reuse, speed or RAM readiness. No AgentLoop/app-tool implementation, model weights, release or tag changes are included.
