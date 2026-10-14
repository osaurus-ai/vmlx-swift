# LFM2-MoE persisted-state arithmetic identity

Changing the router's softmax to the checkpoint's sigmoid, preserving F32 expert bias and matching the reference top-k slot order changes the hidden states saved in KV/recurrent caches. A bundle's weights and token IDs can stay identical across those engine changes. Cache class counts, serializer format and weights fingerprints alone therefore do not distinguish an old incompatible cache.

`LanguageModel.cacheComputationIdentity` is optional and defaults to nil. The actual `LFM2MoEModel` declares `lfm2-moe:sigmoid-f32-bias-positive-topk-v1`, independent of its directory or display name. This applies to both ordinary affine and JANGH installation of that architecture. Other models retain existing keys. Equivalent scheduling or kernel speedups must keep this identity stable; bump it when the model's state-producing arithmetic changes.

`ModelContainer` captures the identity from its model and appends it in both synchronous and asynchronous coordinator setup. It also supplies it to `ModelCacheTopologySnapshot.topologyTags`, which Osaurus already includes in its cache coordinator namespace. Both native setup paths use the same helper and avoid repeating an exact matching pipe-delimited computation component; distinct legacy tags remain intact while the actual current tag is added. The optional snapshot field preserves decoding of older telemetry. Native before/after cache-transition telemetry carries the same identity. This is a computation contract, not a fake `DiskCacheStateProviding` serialization contract, and it does not enable additional payloads, change cache dtype or change cache topology policy.

The coordinator forwards the scoped model key to its paged, disk and recurrent state stores. Old files remain in their namespace; the first request under corrected arithmetic misses them and publishes corrected state. Subsequent requests can reuse the corrected namespace. The change does not clear cache directories or invalidate every model globally.

Low-level callers that directly inject a `CacheCoordinator` instead of using `ModelContainer` own their explicit namespace. Such diagnostic tools must include the actual model's computation identity if they persist shared caches; ordinary Osaurus paths use container setup plus the topology snapshot.

## Regression evidence

An API-compatible CPU regression on pre-fix engine `1a1d32d31a5111b533e0ad36d4a36b34b2fabf2b` reproduced the collision: unchanged key/tokens returned the same disk hash `9ab3199445f987c1569c690ece3bd4bc` and the test failed four assertions.

`LFMCacheComputationIdentityTests` covers:

- Actual LFM construction with an unrelated basename: topology identity and sync/async coordinator keys separate the legacy namespace.
- Unchanged model with an LFM-looking basename: keys remain unchanged.
- Missing identity in old Codable topology telemetry and identity in before/after transition round-trips.
- A CPU-only disk fixture: corrected namespace misses an old record, stores/reopens its own record, and leaves the old record retrievable.

These are component proofs. Full app proof still requires actual old-cache miss/new-cache hit, sustained speed/RAM and multiturn/tool/cache evidence on the final integrated pin. No sampler, seed, kernel arithmetic, cache dtype or eviction policy is modified here.
