---
name: no-driving-turns
enabled: true
event: bash
action: block
pattern: codex\s+exec
---

**Do not drive chat turns to "prove" things. Eric has said no.**

Spawning Codex to click through the app and send prompts burns 40-80k tokens
per run, takes minutes, and repeatedly produced nothing usable — six attempts
at a cold-load chip all returned warm turns because the app warms the model on
window open. Eric's words: *"i said no turns bs"*.

Read the instrumentation instead. It is already there and it is decisive:

- `VMLX_CACHE_FETCH_TRACE=1` — prints raw pre-hash salt components, store
  hashes, `SKIP validated`, and `noRow`. This single trace explained the cache
  "plateau" that six driven turns had failed to explain.
- `OSAURUS_TTFT_TRACE=1` → `/tmp/osaurus_ttft_trace.log` — per-phase TTFT
  breakdown. This is what showed `load_container_done = 0.0 ms` for a 17 GB
  model and corrected a wrong root-cause claim.
- The app's own log in `$OSAURUS_TEST_ROOT` — model load, mmap, quant walk.
- The config JSON on disk — what actually persisted.
- The cache index SQLite — entries, sizes, `created_at`.

A log line with the real number beats a screenshot of a number, and costs
nothing.

If a GUI action is genuinely the only way (clicking a control that has no
other trigger), do the SMALLEST possible one and say why the instrumentation
could not answer it. Do not run multi-turn conversations.
