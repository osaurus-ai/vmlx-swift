---
name: harness-proof
enabled: true
event: stop
pattern: .*
action: warn
---

**Did you prove it in the running harness — visually, with variation?**

Exempt only if this turn changed no osaurus/vmlx behaviour (docs, notes, pure
research). Say so in one line and move on.

Otherwise every behaviour claim needs proof from the RUNNING app. Not curl.
Not the HTTP API. Not RunBench. Not "the unit test passes" — a test proves
code is CORRECT, never that it RUNS. Two migrations and a disabled cache tier
shipped this session precisely because something was correct and unreachable.

**Get that proof from instrumentation, not by driving chat turns.** The traces
answer more, faster, and exactly: `VMLX_CACHE_FETCH_TRACE=1` for salts, store
hashes and `SKIP validated`; `OSAURUS_TTFT_TRACE=1` for per-phase TTFT; the
app log; the config JSON; the cache index. Driving turns is banned — see
`no-driving-turns`.

Per turn, the proof must VARY — a repeated identical prompt proves caching, not
behaviour:

- **Multimodality varies per turn**: text → image → video → audio, mixed with
  tools in the same turn. Named families: qwen / muse / gemma VL for video and
  image, gemma-E2B for audio.
- **Reasoning effort changes mid-conversation**, not just between runs. Watch
  what that does to the cached prefix: if the effort directive sits above the
  static/dynamic boundary, changing it invalidates the whole prefix and forces
  a full re-prefill. Measure the cost; do not assume it is cheap.
- **Long context**: show sustained tok/s as the conversation grows, not one
  short turn. Speed must not degrade as context grows.

Screenshot with `screencapture -l <windowID>` — window-scoped, never
full-screen (a full-screen grab once photographed Eric's API tokens).

State plainly which of these you did and which you did NOT. A gap named is
fine; a gap glossed over is not.
