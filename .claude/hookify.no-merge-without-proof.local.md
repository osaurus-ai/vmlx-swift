---
name: no-merge-without-proof
enabled: true
event: bash
action: block
pattern: gh\s+pr\s+merge
---

**Never merge without full proof.**

Green CI is not proof. CI runs unit tests, and a unit test proves code is
CORRECT, not that it RUNS. This session merged work whose migration had zero
call sites — all checks green, feature completely inert.

Before merging, you must have:

1. **Live visual proof** from the running dev app for every behaviour the PR
   changes — the GUI, not curl, not RunBench.
2. **Variation** in that proof: multimodality varying per turn, reasoning
   effort changed mid-conversation, sustained multiturn rather than one prompt.
3. **Measured numbers** where the PR claims an effect — before/after, not
   "it works now".
4. Every issue found during proving either **fixed**, or stated plainly as
   still open in the merge message.

If you cannot say which live run proved this PR, you have not proven it.

If proof genuinely exists, re-run this command and say in one line what the
proof was — do not merge silently past this.
