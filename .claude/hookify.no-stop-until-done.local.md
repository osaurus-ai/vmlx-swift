---
name: no-stop-until-done
enabled: true
event: stop
action: block
pattern: .*
conditions:
  - field: reason
    operator: regex_match
    pattern: .*
---

**STOP BLOCKED — is the ORIGINAL request actually finished?**

Reporting status is NOT completion. Analyzing is NOT completion.

Self-verify before you try to stop again:

- **(a)** Every single item the user asked for is DONE — not started, not scoped, not analyzed. Re-read the original request and check each item off.
- **(b)** If something crashed, failed, or regressed: it is FIXED and RE-TESTED. A diagnosis is not a fix.
- **(c)** ZERO handoffs. No "next I'll...", no "next steps", no "remaining work", no checkpointing back to Eric.
- **(d)** Writing code without running it is NOT done. Editing a file is not proof it works.

If you are genuinely blocked, it must be an **external** blocker only Eric can supply (a credential, a physical device, a decision he alone owns). State it in ONE line and nothing else.

Otherwise: keep working. Pick the next unfinished item and do it now.
