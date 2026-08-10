---
name: document-every-issue
enabled: true
event: stop
pattern: .*
---

📓 **Write it down. If it is not in the doc, it will be re-derived from scratch — or lost.**

You have re-discovered the same bugs more than once because the finding lived only in a chat window.

**Before stopping, update the durable record:**

- **`/Users/eric/cc/RELEASE_GATE.md`** — the launch gate. Every item PASS / FAIL / BLOCKED **with evidence**.
  Add any new defect with: the **verbatim** broken output, exactly **where** it broke, and whether it
  **REPRODUCES**. Add anything you did **NOT** test to the "NOT TESTED — no proof whatsoever" section. An untested
  row silently reads as "fine" — spell out that it is unproven.
- **Task list** (TaskCreate/TaskUpdate) — one task per real issue, with the repro shape.
- **Memory** (`~/.claude/projects/.../memory/`) — only for durable, cross-session facts: a root cause, a trap that
  fooled you, a proof recipe. Include *why*, not just *what*. Link related notes with `[[name]]`.
- **The wiki** (`wiki remember "<title>" "<body>"`) — only if the fact is useful **from a different project's
  session**.

**Also record what you got WRONG.** The retractions are as valuable as the findings:
- "+33% / +72% compiled decode" — measured with the compile gate **closed**; meaningless.
- "the bundles are patched" — they had been **silently re-downloaded**.
- "compiled decode does nothing" — measured with the gate closed **twice**.
A retraction that is not written down gets repeated as fact.

**And put codex to work on the record, not just the code:** have it produce the in-depth problem analysis, the
proposed fix with `file:line`, the **risk** ("what could this newly BREAK?"), and the **test plan** — then fold
that into the doc. Its analysis is worth more when it survives the session.
