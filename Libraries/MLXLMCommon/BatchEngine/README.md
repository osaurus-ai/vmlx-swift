# Hybrid boundary capture for restored media turns

An image can remain attached to the full conversation after a validated cache
restore has consumed all its placeholder tokens. The remaining input passed to
`prepare` is then text-only. Hybrid inline boundary capture must test that
remaining input, while preserving the original input for cache identity.

Testing the full conversation's media flag skipped the existing inline SSM
capture on every such turn. Finalization subsequently rebuilt the entire
generation-suffix-stripped prefix because recurrent state cannot be trimmed.
One real Osaurus AR request restored 7,923 tokens and answered correctly, then
re-derived an 8,043-token boundary in 9,642.513 ms. The two recorded SSD writes
took 66 and 60 ms; the re-derive is a separate inference-cache phase.

The eligibility guard now uses the prepared suffix's media flag. Inputs whose
cache key is only known after model preparation remain excluded. Existing
splitting, SSM snapshots, media-salted keys, and fallback stores are unchanged.
No AgentLoop, orchestration, or in-app tool implementation changes are involved.

`BatchWarmMediaBoundaryTests` exercises the real `BatchEngine.submit` path with a
deterministic KV/recurrent-state recorder. The warm control restores six tokens
of a thirteen-token media prompt, with an eleven-token canonical boundary. The
baseline forwards the seven new tokens and then replays eleven tokens during
finalization; the candidate forwards only the seven new tokens. Restored KV and
recurrent offsets and the prefix checksum are checked. Cold-media and
changed-image controls preserve the original payload and reject cross-image
reuse. All three tests pass after the change; the baseline warm test fails two
assertions. The recorder generates one token with an expected length stop; it
does not qualify a real language model.

A normal Release dev app on implementation `0927979f` subsequently exercised
three growing AR turns over the same real image history. Answers were correct;
restored prefixes were 8,043, 8,205 and 8,322 tokens. Canonical inline store
boundaries were 8,205, 8,322 and 8,492 tokens, with finalization return times
2.315, 2.107 and 1.985 ms and no boundary re-derive in that run. UI decode rates
were 22.6, 23.9 and 24.0 tokens/s. Different turn lengths and the app's idle-load
policy mean these rows are cache-behavior proof, not a paired speed benchmark.

The same production implementation completed an actual 45-second foreground
sandbox command and restored 12,175 tokens for the correct post-tool answer.
Test-only signing and socket-path repairs were needed before execution; no
sandbox, tool or orchestration implementation was changed. The subsequent
DFlash arithmetic failure and invalid frozen-control AR replay were retained
separately; this boundary guard does not qualify general model quality.

Cold media and post-prepare-key media retain their existing paths. A later cold
AR request still rebuilt a 6,833-token stable-system seed in 9.196 seconds;
this distinct finalization path remains open. The change does not remove those
costs or establish numerical parity for every model family. Selected Swift 6
adjacent tests passed (77 passed, one inventory skip); the full focused suite
and normal review/CI remain separate gates.
