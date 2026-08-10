---
name: merge-needs-live-proof
enabled: true
event: bash
pattern: gh\s+pr\s+merge
---

🚦 **Merging. For each change in this PR — is it LIVE-PROVEN, or merely written?**

Do not merge a runtime/UI change on the strength of green tests. State, per change, which bucket it is in:

- **PROVEN** — seen working in the dev-built app via codex computer-use, on the shape that reproduces the bug
- **NOT EXERCISED** — the code path never ran during the gate (say so out loud; do not let it ride on the coattails
  of the proven ones)
- **PRE-EXISTING** — verified against clean `main` before blaming the branch

**Hard-won lessons this exists to prevent:**

- A gate ran against **the wrong model** (asked for `12B MXFP8`, silently got `12B MXFP4`) — every Top-Pick claim
  from that run was worthless.
- A gate ran against **the wrong app instance** (two osaurus processes) — the evidence was discarded.
- A gate ran against a **binary without the fix** (repin not done / app not rebuilt).
- A **Debug** build was used for perf numbers, making a fast model look 3-4x slow.
- A fix was written that was **dead code on the crashing path** (the `unknownTokenId` fallback) — it would have
  fixed nothing and looked like a fix.

**Also confirm before merging:**
- No AI attribution anywhere in commits/PR body (absolute rule).
- No forced-behavior hacks to fake coherence (forced thinking tags, decode-loop biasing, template coercion,
  synthetic sampling defaults). If you found one, remove it and root-cause the real failure.
