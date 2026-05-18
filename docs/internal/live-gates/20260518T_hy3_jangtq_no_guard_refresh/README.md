# Hy3 JANGTQ No-Guard And Reasoning Refresh

Timestamp: 2026-05-18 16:12 PDT

Model:

`/Users/eric/models/JANGQ/Hy3-preview-JANGTQ`

## `BENCH_NO_GUARD_SAMPLING`

Artifact: `no_guard_sampling.log`

Result:

- Loaded as `Hy3Model` with `hunyuan` tool format and `qwen3` reasoning stamp.
- Greedy row used `temp=0`, `topP=1`, `topK=0`, `minP=0`, `rep=nil`.
- Output stopped normally: 12 tokens, 23.3 tok/s, no loop, no BOS repetition,
  no marker leak, visible greeting.
- The generic thinking-on row produced coherent visible output with
  `rep=1.0`, but the prompt tail still rendered `reasoning_effort:no_think`.
  This proves the generic `enable_thinking=true` knob is not Hy3's native
  reasoning control.

## `BENCH_REASONING_TURN_MATRIX`

Artifact: `reasoning_turn_matrix.log`

Result:

- Loaded as `Hy3Model`; RSS delta at load was about 6414 MiB.
- Bundle/default sampling resolved to `temp=0.900`, `topP=1.000`,
  `topK=-1`, `minP=0.000`, `rep=nil`.
- Multi-turn recall stayed coherent:
  - saved `copper-lantern`;
  - recalled `copper-lantern`;
  - answered `blue`;
  - answered `20`.
- Plain `enable_thinking=true` remained not-template-active for Hy3.
- Native `reasoning_effort=low` and `reasoning_effort=high` routed real
  reasoning deltas while keeping visible content clean and stopping normally.
- `reasoning_effort=medium` and `max` stopped normally with visible content and
  no reasoning deltas on this prompt.
- Cache stats: paged misses 8, disk hits 1, disk misses 36, disk stores 24,
  SSM hits/misses/rederives all 0 for this non-hybrid snapshot.

Boundary:

This is vmlx runtime proof, not Osaurus app/API production-clear. It confirms
that Osaurus must keep Hy3 on the native `reasoning_effort` path rather than a
generic `enable_thinking` toggle. Osaurus still needs route, UI, cache, memory,
tool-result, and saved-setting carryover artifacts.
