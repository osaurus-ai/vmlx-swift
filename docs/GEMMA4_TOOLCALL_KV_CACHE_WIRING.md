# Gemma-4 tool-call KV-cache wiring (osaurus ↔ vmlx-swift)

Status: **root cause fixed** — osaurus-side in `osaurus-ai/osaurus#1525`
(`e5419a3b`), vmlx-side contract now locked by regression tests in this repo.

## Symptom

On `gemma-4-*-qat-*mxfp4` (and `jang_4m`) models, an agent loop that called a
tool (classically `capabilities_discover`) showed the prefill counter **reset
to 0** on the very next iteration — the entire prompt re-prefilled instead of
extending the cached KV prefix. TTFT spiked after every tool call; long agent
loops became quadratic.

## Root cause — the full chain

The Gemma-4 prompt renders its tool declarations at the **front of the system
turn**, so the system turn is the segment the KV cache reuses across loop
iterations. Two things conspired to mutate that segment between iterations:

1. **osaurus `ChatToolChoicePolicy.finalizingPostToolChoice`** had a special
   case: for `gemma-4 + qat + (mxfp4 | jang_4m)`, on any iteration whose last
   message was a `tool` result, it force-flipped `tool_choice` to `.none`.
2. **osaurus `ModelRuntime.makeTokenizerTools`** returns `nil` when
   `tool_choice == .none` (`ModelRuntime.swift:2431`), stripping the entire
   tools list passed to the tokenizer.
3. **vmlx `Libraries/MLXLMCommon/ChatTemplates/Gemma4WithTools.jinja:42-49`**
   only renders the `<|tool>declaration:…` block when `tools` is truthy. With
   the tools stripped, that block **vanishes from the front of the system
   turn**.

Net: iteration N (tool call) and iteration N+1 (tool result) rendered
**different system turns**. The cached prefix diverged immediately after
`<bos>`, so the runtime re-prefilled from token ~0 — the "kv cache set to 0"
symptom.

A second, related prefix-buster: transient `[System Notice]` lines (tool-budget
nudges, etc.) were appended as a trailing **user** turn. Templates that anchor
the assistant reasoning rail on "the last user query" (e.g. Qwen3.x
`last_query_index`) then re-rendered the cached assistant tool-call turn
*without* its `<think>` scaffold, diverging from the stored KV prefix.

## The fix

### osaurus side — `osaurus-ai/osaurus#1525` (`e5419a3b`), already on main

- **Deleted `finalizingPostToolChoice` entirely.** No more gemma-4 tool_choice
  flip; the loop keeps the requested `tool_choice` byte-stable across every
  iteration (`HTTPHandler.swift` `modelStep` now passes `resolvedToolChoice`
  unchanged and `tools: tools.isEmpty ? nil : tools`).
- **Added `AgentLoopBudget.appendingTransientNotices`.** When the iteration
  already ends in a tool result, transient notices ride as additional
  **tool-role** environment feedback (sharing the result's `tool_call_id`)
  rather than a trailing user turn, so the last-query anchor — and the reused
  KV prefix — stays put. Falls back to a trailing user turn only when there is
  no tool result to attach to (e.g. the empty-turn nudge).

### vmlx-swift side — this repo (contract lock, no template behavior change)

The `Gemma4WithTools.jinja` template is the *other half* of the wiring: it is
correct **as long as the caller holds `tools` + `tool_choice` byte-stable**,
which is exactly the invariant #1525 now guarantees. To keep a future caller
(or a future template edit) from silently reintroducing the bug, two
regression tests in
`Tests/MLXLMTests/Gemma4ChatTemplateProbeTests.swift` pin the contract — pure
template renders, no model weights, CI-safe:

- `testGemma4WithToolsSystemPrefixStableAcrossToolResultTurn` — renders the
  same conversation at the tool-call turn and at the tool-result
  continuation; asserts the system turn (system content + `<|tool>declaration`
  block) is **byte-identical** and remains a literal prefix of the
  continuation prompt → KV prefix is reused.
- `testGemma4WithToolsDroppingToolsBustsSystemPrefix` — asserts the failure
  mode directly: drop the tools list and the with-tools system turn is no
  longer a prefix of the prompt. Documents *why* the caller must never strip
  tools on a tool-result turn.

## Invariant for any caller driving Gemma-4 tools

> Across all iterations of one agent loop, keep `tools` and `tool_choice`
> byte-stable, and deliver per-iteration nudges as tool-role feedback (not a
> trailing user turn) whenever the iteration ends in a tool result. The Gemma-4
> system turn must be byte-identical iteration-to-iteration; anything that
> mutates it re-prefills the whole context.

## References

- osaurus: `Packages/OsaurusCore/Services/Chat/ChatToolChoicePolicy.swift`,
  `Packages/OsaurusCore/Services/Chat/AgentToolLoop.swift`,
  `Packages/OsaurusCore/Networking/HTTPHandler.swift`,
  `Packages/OsaurusCore/Services/ModelRuntime.swift`.
- vmlx: `Libraries/MLXLMCommon/ChatTemplates/Gemma4WithTools.jinja`,
  `Tests/MLXLMTests/Gemma4ChatTemplateProbeTests.swift`.
- osaurus PR: https://github.com/osaurus-ai/osaurus/pull/1525
