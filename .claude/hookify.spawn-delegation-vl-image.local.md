---
name: spawn-delegation-vl-image
enabled: true
event: all
pattern: (spawn|Spawn|delegat|Delegat|SubagentKind|subagent|ImageGeneration|image edit|imageEdit|mflux|OCR|ocr|VLM|vision|visionModel)
---

🧩 **Spawn / delegation / VL / image — each of these is its own bug surface. Prove them INDIVIDUALLY.**

None of this is proven by "chat works". Every hop can silently drop something, and it has.

**SPAWN / AGENT DELEGATION — trace and PROVE every hop:**
1. The spawn/delegate **model is settable** in the UI, and the one you set is the one that runs.
2. The **prompt passed TO the spawned model is visible** in the harness — read it VERBATIM.
3. The spawned model **actually does the job** (not an empty/■hallucinated result).
4. The spawned agent's **TOOLS execute**. ⚠️ *Known shipped bug*: spawned children ran **text-only** because
   `TextSubagentKind` never resolved the persona's tools (child request had `tools: nil`). A source audit says a
   spawned agent **can still silently lose its configured tools** — treat as UNFIXED until seen live.
5. The child's **result is handed BACK** to the orchestrator and **actually USED** in its final answer.
6. **Cross-model** delegation (gemma orchestrator → ornith child) works, and does not crash on GPU residency
   handoff (past SIGSEGV: concurrent residency; unload must DRAIN the GPU before freeing weights).
7. Delegation to a **cloud** model — does the warm pass / preload apply, or is it nonsense there?
8. ⚠️ Known: **different-model delegation cold-loads on EVERY handoff** (no warm reuse).

**VL / VISION — an actual image must go through:**
- Attach a **real image**; the model must describe it **correctly** (use unambiguous content — a hallucinated
  description that sounds plausible is a FAIL).
- Two images in one turn. Image + tools together. Image in a **multiturn** conversation (turn 3 still sees it).
- A non-vision model handed an image must fail **gracefully**, with a clear message.
- OCR-style task: read text out of a screenshot **verbatim**.
- An **OCR spawn/delegation job** (driver model → VL child → synthesis) — prove the image reaches the child.

**IMAGE GEN / EDIT (custom mflux runtime):**
- Generate: a **real PNG appears**, and it is **prompt-relevant** (and prompt-**sensitive**: a different prompt
  gives a different image).
- **Edit**: supply an image + instruction; the edited image reflects the instruction.
- Chained **gen → edit**.
- An **agent can spawn/delegate an image job** and the image comes **back into the chat**.
- ⚠️ **Image gen/edit BYPASS memory admission entirely** — no budget preflight. A big image model can page the
  machine.
- GPU **drain** on chat→image and image→model-load handoff (two symmetric points; missing drain = crash).
- Progress/spinner shown; UI not frozen; the chat model still works afterwards.

**Do not fold these into one "it all works" claim.** Report each row PASS / FAIL / NOT TESTED.
