---
name: settings-parity
enabled: true
event: stop
pattern: .*
action: warn
---

**A setting that saves but does not take effect is the default failure here.**

Only applies if this turn touched settings, generation params, context sizing,
MTP/speculative toggles, or anything the user can change in the UI.

This pattern has now been found four separate times in this codebase:

- `ChatConfiguration.contextLength` — a FALLBACK, last in the chain, inert for
  every model that declares its own window.
- `migrateToCurrentSchema()` — zero call sites.
- Sampler defaults — the user's Settings sit BEHIND model-shipped defaults, so
  temperature / topP / topK are likely inert for most bundles. **Still open.**
- The `%.1f` field that rewrote 0.005 to 0, which the resolver then ignored.

So for anything a user can change, verify all four:

1. It **persists** to disk with the value they typed (check the JSON, not the
   field).
2. It is **read** on the path that runs — trace to the object that enforces it,
   not the struct that holds it.
3. It **wins** against defaults it should outrank, and loses to ones it should
   not. Write down the precedence order and check it.
4. The UI **displays** the value actually in force, including any downstream
   clamp. Displaying the requested value while the engine holds another is a
   lie the user cannot detect.

Also check the API-server path, not just chat — a setting wired only into the
chat UI is half-wired.

MTP / speculative toggles: the gate lives in vmlx (`canUseNativeMTP`) and there
are multiple dispatch sites. Verify the toggle agrees with the gate at EVERY
one, and that the chat path (BatchEngine) is included — it has been missed.
