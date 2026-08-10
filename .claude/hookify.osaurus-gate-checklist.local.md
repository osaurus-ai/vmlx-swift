---
name: osaurus-gate-checklist
enabled: true
event: bash
pattern: codex\s+exec.*(computer.use|osaurus|GUI|visual|live)
---

📋 **You're driving the live osaurus harness. Make codex look for ALL of it — not just the one thing you fixed.**

Every past regression was found by *reading the output closely*, not by the app "working". Instruct codex to
quote every defect **VERBATIM, character for character** — a paraphrased defect destroys the evidence.

**The standing watchlist (do not omit any of these):**

- **Leaking** — `<|...|>`, `<channel|>`, `<|channel>`, `<tool_call>`, `</tool_call>`, `<think>`, `</think>`,
  `<start_of_turn>`, `<end_of_turn>`, bare `thought` / `channel`, orphan closing tags, raw template tokens
- **Weird output** — gibberish, garbled/broken characters, mojibake, stray CJK or random unicode in English,
  token-corruption misspellings (e.g. `BANN划CHECK`), internal self-correction prose bleeding into the answer
- **Looping / repetition** — the same line 3+ times; agent loops re-calling an already-satisfied tool
- **Incoherency** — losing the thread; failing to recall turn 1 (**that is a CACHE/CONTEXT failure, report it loudly**)
- **Truncation / empty replies** — cut off mid-word; empty reasoning-only turns
- **Settings not taking effect** — toggles that change nothing (reasoning ON/OFF, tools ON/OFF, RAM safety mode).
  ⚠️ Known real bugs: **tools have fired while tools were OFF**; a **thinking block appeared with Thinking OFF**
- **Reasoning** — block opens AND closes; contents stay OUT of the visible answer
- **Tool usage** — card appears, tool actually RUNS, and the final answer genuinely USES the result (not hallucinated)
- **Cache warmup** — does it actually run on load / on model switch? (`ModelWarmup` was found to be **dead code** — no production call sites)
- **Spawn / delegation** — the spawn model is settable; the prompt passed to the child is visible; the child does
  the job; the child's **tools execute**; the result is handed BACK and USED by the orchestrator
- **Model switch** — A→B→A; does context survive? does it hang?
- **Determinism** — same prompt twice: identical or not?

**Also tell codex:**
- State the model it used **VERBATIM**. Never let it silently substitute a different quant.
- An **intermittent** leak seen once is a FAIL. "It didn't reproduce" ≠ clean.
