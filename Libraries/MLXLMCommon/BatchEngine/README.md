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

This is a source/unit checkpoint. A dev app on this change must still exercise
real media history, growing turns, cache restore, and finalization before merge
readiness. The single measured baseline row is not a paired performance result.
Cold media and post-prepare-key media retain their existing paths; this change
does not claim to remove their finalization costs or establish numerical parity
for every model family.
