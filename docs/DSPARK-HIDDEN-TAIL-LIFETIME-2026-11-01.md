# Bound retained shared-KV and draft-window graphs

Long DeepSeek-V4.1 AR answers failed Metal shared-event creation at exactly 1,636 output tokens, both after DSpark attachment and in a fresh process without a drafter. Private lifetime instrumentation found 65,505 live shared events before allocation failed, while command buffers stayed bounded at 24 or fewer. Events returned to zero when the request cache was released. A diagnostic with per-layer submission disabled completed 5,770 tokens normally, but disabling that speed optimization is not the fix.

## Shared latent KV

DeepSeek attention supplies the same latent as K and V, then consumes only K. The generic rotating cache independently builds a lazy V update chain. Because attention never evaluates that branch, it retains every old projection row and its completion event. Per-layer asynchronous submission turns that into roughly one retained event per layer per token.

`RotatingKVCache.updateSharedKV` keeps the unchanged K update and aligns V with the updated K through a fresh array handle. The separate unused V chain is released. Its explicit contract requires shared K/V history; ordinary distinct-K/V updates are unchanged. The DeepSeek companion cache preserves its existing local-window compaction. Normal attention, cache-only prefill and speculative prefix commits use the shared path. Custom cache implementations asynchronously evaluate their returned V branch instead. No host wait, sampler change, output clamp, cache-format migration or layer-submission disable is added.

A minimal test using the real rotating-cache class and 40 layer submissions reproduces the same allocation error in 5.78 seconds without loading a model. The shared update completes all 4,096 decode steps across 40 caches. Two CPU controls match paired updates through prefill, growth, rotation, restored state and DeepSeek companion windows. These 3 tests pass in one actual suite. A fresh app build and affected full-model long-AR/performance qualification remain pending.

## Draft-window tail

An attached DSpark head also records bounded hidden rows while a later request selects AR. Joining and slicing chunks bounded visible rows but left a lazy history chain. Scheduling compacted rows with `asyncEval` bounds that separate history without a host wait, preserving positions, values and rollback. It serves DeepSeek-V4.1 and MiniMax-M3.

The tail regression fails on the landed runtime and passes with asynchronous compaction. Earlier retained runs pass 37 tests across 3 focused suites and another 48 tests across 6 related MiniMax/cache suites. The final tail change also passed its 3 new tests. A separately added Swift control passed 4,096 steps with 40 layer submissions. These are distinct runs, not one aggregate invocation. The tail change alone did not prevent the full-model failure; those failed rebuilt-app rows remain retained separately.

## Qualification limits

The layer-submission-off diagnostic is neither a production fix nor speed acceptance: it measured 21.40 tok/s. Its generated module passed 20 unit tests and 130 independent inputs, with 3 docstring-example failures retained separately. Earlier native sampled Adaptive easy-code completed at 33.0 tok/s with passing generated tests and 130 independent checks; that is not a 60 tok/s result or a valid speedup against failed AR. Production model defaults remain unchanged. No release is requested.
