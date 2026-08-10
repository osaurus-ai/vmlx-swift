---
name: finish-after-failure
enabled: true
event: stop
action: block
pattern: .*
conditions:
  - field: reason
    operator: regex_match
    pattern: .*
---

**DID ANYTHING FAIL THIS TURN? THEN YOU MAY NOT STOP.**

Scan what actually happened in this turn:

- Build failed or errored?
- A test failed, hung, or was skipped?
- The app crashed, hung, or produced wrong output?
- A command exited non-zero and you moved on?

If ANY of those happened, the turn is not over. Required sequence:

1. Fix the actual cause — no guessing, no "likely".
2. Rebuild.
3. Re-run the exact thing that failed.
4. Verify it now passes.

A crash report, a stack trace, or a root-cause writeup is a STARTING POINT, not a deliverable. Handing Eric a diagnosis and stopping is the failure mode this rule exists to kill.

Do not stop to ask whether he wants it fixed. Fix it.
