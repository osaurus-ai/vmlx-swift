---
name: no-forced-behavior-fixes
enabled: true
event: file
pattern: (forced|force).*(think|reason|channel)|inject.*(think|reason|thought)|logit.*bias|logitBias|token.*bias|repetitionPenalty\s*=\s*1\.[1-9]|temperature\s*=\s*0\.\d+\s*//\s*(force|fix|workaround)|stripThinking.*insert|append.*</think>|prepend.*<think>
---

🚫 **This looks like a forced-behavior fix. Do not fake coherence.**

**Never** paper over a runtime bug with:
- forced thinking tags, hidden reasoning openers/closers
- decode-loop token biasing / logit biasing
- synthetic temperature / top-p / top-k / repetition-penalty defaults
- prompt or chat-template coercion to suppress a symptom

Sampling defaults must come from the **model bundle's generation config**, unless the user or the API explicitly
overrides them.

**If you find such a guard already in the code: document it, REMOVE it, and root-cause the real failure** — in
template parsing, runtime/decode, cache, kernels, model loading, or API wiring.

**This is not the same as repairing a malformed prompt.** Fixing a chat template that closes a turn and strands
generation role-less is a *correctness* fix: it hands the model a **well-formed turn** instead of a malformed one.
It biases nothing. Know the difference, and be able to explain which one you are doing.

**And the mirror image:** a coherent-but-wrong output is **almost always a runtime/impl bug, not the quant**.
100+ models work at 2-3 bit. Prove it via probing / per-layer comparison before blaming quantization.
