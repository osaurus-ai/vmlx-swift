---
name: cache-percent-enforced
enabled: true
event: stop
pattern: .*
action: warn
---

**Cache work: did you watch the WRITES and the EVICTIONS, against the % cap?**

Only applies if this turn touched cache sizing, storage, eviction, prefix reuse,
or the disk-L2 tier.

The cap is a **percent of the SSD**, not gigabytes. Check the whole chain, not
one link — each of these has already been found broken once:

1. The share is **persisted** (`maxSizePercent` in `server-runtime.json`).
2. Migration actually **runs** on load (`migrateToCurrentSchema` had zero call
   sites and silently did nothing for every updating install).
3. It reaches `CacheCoordinatorConfig.diskCacheMaxGB`.
4. The **host-aware ceiling** is applied to what you DISPLAY, not just what is
   enforced — Settings once promised 372 GB while the engine held 242 GB.
5. A small share is **honoured**, not clamped up by a floor and not silently
   disabling the tier.

Then watch it actually happen:

- **Writes**: is the cache still growing turn over turn? A cache that plateaus
  BELOW its cap while turns run is not "settled", it is suspicious — that exact
  observation is still UNEXPLAINED in the audit doc.
- **Evictions**: measure before/after. Assert entries SURVIVE as well as
  shrink; a cache that refused every write also looks small.
- **Oldest-first across models**, not per-model budgets.

Multimodal keying: every VLM `LMInput` must carry `cacheScopeSalt`. Unsalted
media means same-text/different-image collides and answers fluently about the
WRONG picture. `DeepseekOCRProcessor` was exactly this.

Numbers or it did not happen. "The cap is applied" is not a measurement.
