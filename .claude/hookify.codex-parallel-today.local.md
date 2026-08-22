---
name: codex-parallel-today
enabled: true
event: all
pattern: .*
action: warn
---

**TEMPORARY — today only. Delete this rule at end of day.**

**Are you running Codex gpt-5.6-sol xhigh IN PARALLEL right now?**

Not "spawn one and wait for it". Parallel means: fire Codex on an independent
slice, then **keep working yourself** while it runs, and collect its report when
it lands. If you are sitting idle waiting on a Codex call, you are doing it
wrong. If you did a whole audit/scan/probe alone that Codex could have run
alongside you, you wasted it.

Invocation — never `fast` mode:

```bash
codex exec -m gpt-5.6-sol -c model_reasoning_effort="xhigh" \
  --sandbox danger-full-access -C <repo> "<prompt>" > /tmp/codex-<slice>.log 2>&1 &
```

Run several at once on **disjoint** slices so their findings don't collide.

**Every prompt must be specific and must demand a report back.** A vague prompt
returns a vague answer that costs you a turn to re-ask. Each one states:

1. the exact files/paths/symbols to look at
2. the precise question, with the failure shape you suspect
3. what would prove it either way — a command to run, a test to write, a
   measurement to take
4. **"Report back: what you checked, what you found, the exact file:line for
   every claim, what you ruled OUT and why, what you could not determine, and
   what you would look at next."**

Then use what comes back — cross-check its file:line claims before acting on
them. Codex is a second perspective, not an oracle; it has been confidently
wrong here before. Verify, then act.

Good parallel slices for the current campaign: SSD block prefix/suffix
best-match reuse, MoE + SwiGLU JIT-compile opportunities, upstream mlx-swift /
mlx-swift-examples / mlx_vlm commits worth porting, max-context enforcement and
compaction trigger timing, SSM/TQ q4 correctness, tool-call JSON vs XML
parsing.
