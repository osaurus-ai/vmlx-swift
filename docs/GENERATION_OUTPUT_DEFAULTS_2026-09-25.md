# Bundle output-token defaults

`generation_config.json` and JANG `chat.sampling_defaults` accept `max_new_tokens` and its publisher alias `max_tokens`. A valid `max_new_tokens` takes precedence. A cap must be a positive integral JSON number representable as Swift Int; null, booleans, strings, zero, negative, fractional and overflowing values are absent. An invalid canonical value may fall back to a valid alias. `max_length` and context-window size are not output-token caps.

Encoding uses canonical `max_new_tokens`. Explicit generation parameters continue to override adopted defaults. No sampler or reasoning-mode defaults are introduced.

Qualification: exact-source CPU parser probe covers 17 boundary rows across decoder/dictionary/app readers and round trips. Full package tests and Osaurus runtime/UI proof are tracked under `/Users/eric/vmlx-private-evidence/raptor06-speed-2026-09-23/parallel-closeout-2026-09-25/generation/`; CPU parser evidence alone is not runtime qualification.
