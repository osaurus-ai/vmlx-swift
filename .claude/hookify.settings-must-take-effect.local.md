---
name: settings-must-take-effect
enabled: true
event: all
pattern: (memorySafety|MemorySafety|RAM safety|ramSafety|compiledDecode|Decode Perf|CacheStoreBudget|GPUMemoryBudget|checkRAMFeasibility|InferenceFeatureFlags|ServerRuntimeSettings)
---

⚙️ **A setting the user can move MUST change what the runtime actually does — or it is a lie.**

The RAM / memory-safety slider is currently **decorative as a guarantee** (audited, source-cited):

- GPU working-set excess is **warning-only** — the user is still allowed to send, and the machine pages.
- `checkRAMFeasibility` judges against **physical RAM** and is **advisory** for mmap loads.
- Strict / "fail closed when estimate unknown" **never receives a working-set estimate**, so its blocking issues
  are ignored.
- **Image generation and image edit bypass memory admission entirely** — no budget preflight at all.
- Changing the setting **mid-session does not refresh resident models**.
- ~9 controls are persisted and read but **never applied** (Prefill/Completion Batch Size, SMELT Mode, Stored KV
  Codec, "Fail Closed When Estimate Is Unknown", Max Concurrent Sequences, Keep Draft Cache Separate, …).

**This is not cosmetic — it is the likely root cause of the hangs.** 20 of 25 Sentry issues are
"App Hanging ≥3s" with *scattered* culprits (fonts, `NSWindow.init`, `copyTurnContent`). A real user on an
**8 GB M1 Air with 97 MB free** hung **10 times in 7 minutes**. Those are **page-fault stalls on a thrashing
machine**, not blocking I/O — because a model that does not fit was admitted anyway.

**Rules when touching this:**

1. **Budget against the GPU working set**, not physical RAM. Loading past the Metal working set does **not** error —
   macOS just pages, and the user experiences it as a hang (~1 char/10s), never as an error.
2. The refusal must be **REACHED**, not merely correct. We have twice shipped a locally-correct guard the system
   never reached (an actor-reentrant check-then-act; a host-side counter inside a compiled graph body that
   `compile()` replays straight past). **Prove the guard fires.**
3. **A gate that is too strict is as bad as one that is too loose.** A hard gate would newly REFUSE models that
   work today (a 94 GB model runs fine on a 128 GB Mac but exceeds Safe Auto's 89.6 GB budget). Before flipping
   one, run the live matrix: just below / at / just above each ceiling, per mode, for dense + routed JANG + VLM +
   image.
4. If a control cannot be honestly wired before launch, **remove it from the UI** (keep the Codable/API field for
   compatibility). Shipping a settings panel full of controls that do nothing is a credibility problem.
5. **Prove it live**: change the setting and demonstrate an OBSERVABLE difference in what is admitted/refused —
   in the app, not in a unit test.
