---
name: no-unproven-claims
enabled: true
event: stop
action: block
pattern: .*
conditions:
  - field: reason
    operator: regex_match
    pattern: .*
---

**PROOF GATE — did you RUN it and SEE it work?**

Eric's standing rule: nothing is "done" without LIVE proof in the running dev app GUI.

Not proof:
- curl / HTTP / any API call (explicitly banned)
- unit tests passing on their own
- "the build succeeded"
- reading the diff and reasoning that it should work

Proof: the dev-built app is running, you drove the real UI, and you SAW the correct behaviour — multiturn where the feature is multiturn.

Before stopping, answer honestly:
1. Did I launch the app?
2. Did I exercise the exact changed path?
3. Did I observe the result, not infer it?

If any answer is no, the proof gate was NOT run. Say that in one line, then go run it. Do not stop, and do not describe unproven work as fixed, working, or complete.
