---
name: osaurus-live-visual-proof
enabled: true
event: stop
pattern: .*
---

🛑 **Before you stop: is the claim PROVEN, or just believed?**

For any osaurus / vmlx runtime claim, **proof means one thing only**:

> **A live visual multiturn run in the DEV-BUILT osaurus app, driven by codex computer-use.**

CLI, HTTP API, curl, unit tests, `swift test`, and "it compiles" are **DIAGNOSIS ONLY — never proof.**
The API path is not the same code as the app harness. It has repeatedly said "fine" while the app was broken.

**Answer honestly before stopping:**

- [ ] Did I rebuild the dev app **after** the last merge/repin, and confirm the change is actually **inside the binary** (`strings … | grep <new symbol>`)? A stale binary reporting green is the exact way we've shipped false confidence before.
- [ ] Did codex drive the **real GUI** and read every streamed reply **character by character**?
- [ ] Did I test the **model that actually ships** (a Top Pick), not whatever the picker happened to offer? Codex has silently substituted a different quant before — make it state the model **verbatim**.
- [ ] If I'm claiming something is fixed, did I run **the exact shape that reproduces the bug** (e.g. tool-call → tool result → follow-up), not a generic happy path?
- [ ] If a defect was **intermittent**, do I understand that **one clean pass proves nothing**?

**If any box is unticked, say so plainly instead of claiming it's fixed.**
"Tests pass" / "it builds" / "the API returned 200" are not answers.
