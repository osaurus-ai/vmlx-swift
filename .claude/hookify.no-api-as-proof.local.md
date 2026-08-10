---
name: no-api-as-proof
enabled: true
event: bash
pattern: curl.*(1337|/v1/chat|/v1/completions|/agents/|localhost:\d+/v1)
---

⚠️ **You are hitting the osaurus HTTP API. That is NOT proof.**

The API surface is **not the same code path** as the app harness. It has returned clean results while
the GUI was visibly broken (leaked markup, loops, dead tools, wrong model loaded).

**This is fine for DIAGNOSIS. It is never evidence that a bug is fixed.**

If you are trying to demonstrate that something works for the user:

1. Build the dev app (Release) from the PR branch.
2. Launch it — and check there is **exactly one instance** running (`pgrep -f osaurus.app | wc -l`).
   Codex has driven the *wrong* osaurus instance before and reported on a stale app.
3. Drive it with **codex computer-use** and read the actual rendered output.

If the GUI path is genuinely blocked, **say it is blocked** — do not silently downgrade to curl and
present the result as if it were the same thing.
