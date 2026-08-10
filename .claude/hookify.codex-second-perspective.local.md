---
name: codex-second-perspective
enabled: true
event: stop
pattern: .*
---

🧠 **Did you get a SECOND perspective, or just trust yourself?**

Run **codex `gpt-5.6-sol` at high reasoning effort, in parallel**, on any non-trivial diagnosis or fix. It is a
separate agent, uncontaminated by your context, and it has repeatedly caught what you could not see.

**It has already saved you from shipping:**
- A **TOCTOU** residency guard (correct, but the system never reached it).
- A **decorative precondition** inside a compiled graph body that `compile()` replays right past — you were about
  to call it a safety guard.
- A fix that was **dead code on the crashing path** (the `unknownTokenId` fallback would have fixed nothing).

**Use it two ways, in parallel — source audit does not need the GPU, so it costs you nothing:**

1. **ADVERSARIAL**: give it your claim and tell it its ONLY job is to **REFUTE** it. Not to agree, not to
   summarize approvingly — to find where you are wrong. Demand `file:line` and `REAL BUG` vs `SAFE`, and demand
   it say **UNPROVEN** rather than speculate.
2. **LIVE VISUAL**: computer-use driving the real dev-built app, reporting defects **VERBATIM**.

**And doubt yourself first.** Before asserting a cause: did you *verify* it, or does it merely sound right?
You have been wrong today about: the cause of the compiled-decode sign flip, "compiled decode does nothing"
(you measured with the gate closed — **twice**), and "the bundles are patched" (they had been silently
re-downloaded). **Verify before asserting. No assumptions. No false positives.**
