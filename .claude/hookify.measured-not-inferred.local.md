---
name: measured-not-inferred
enabled: true
event: stop
pattern: .*
action: warn
---

**Was the number measured, or inferred?**

Only applies to claims you are about to make about speed, cache behaviour or
correctness.

- Speed: hot **ABAB**, never A-then-B, and log `PhysMem` before each leg. A
  saturated box has fabricated a 29% regression here before.
- Long context: report the **curve** (2k/8k/16k/32k), not one point, and say
  whether decay is depth-dependent or flat. DSV4 turned out flat when it
  "obviously" sagged.
- One sample is not a finding. Three at a fixed condition, median reported.
- Suspect the harness first when a result looks alarming or reassuring. Assert
  a non-empty baseline so an empty-vs-empty comparison cannot read as PASS.
- Test a size that does NOT divide evenly by the step/page/block.
- If you did not measure it, say "unverified" — never "fixed" or "working".
