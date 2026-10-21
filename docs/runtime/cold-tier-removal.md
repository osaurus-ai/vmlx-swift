# Cold-tier removal

Draft #632 removes the JangPress/MLXPress cold-weight subsystem. It removes router observation, cold-page sweeps, prompt-token host readbacks used only by the embed tier, loader activation and status/policy APIs, separate mmap/embed caches, Mach purgeable weight copies, experimental prestack/overlay creation, and the standalone MLXPress products. The Osaurus companion removes the corresponding setting and status response fields. Persisted old settings are ignored, not reactivated.

Canonical model loading remains: original bundle identity, safetensors mmap, explicitly authorized alignment repair, model residency leases, affine/JANGH/JANGTQ matmul, Hadamard/trellis transforms and active routed-tensor reads. `MappedSafetensorsShard` extracts the read-only shard/index/range reader used by affine streaming; it has no eviction/advice API. `LegacyRoutedTensorLayout` only recognizes already-existing stacked weight layouts; it cannot create an overlay. Historic MLXPress-prefixed profiling and kernel-selection names are not the retired cold-tier subsystem and retain their existing behavior.

The runtime's typed `loadModel(...loadConfiguration:)` now returns `ModelContext` directly. Cold-tier tuple handles and legacy activation overloads are removed. Repository benchmark clients migrate to that API. This is a source-breaking removal for external callers of the retired API/products; ordinary model-container loading remains available.

## Validation and boundaries

- MLXVLM and RunBench compiled after activation/API/overlay removal. Subsequent Mach-copy removal is undergoing its own compile and focused tests.
- Focused coverage includes legacy-setting migration, load configuration, canonical shard reads and explicitly authorized storage repair. Tests of deleted cold-tier behavior are retired with those APIs; quantized compute/cache tests are retained.
- Final Osaurus build, affine/JANGH connected chat and SSD restore checks remain pending. Do not merge this draft until that evidence is attached.
- No speed or RAM reduction is claimed from deletion alone. Most production paths already disabled cold-tier activation. The explicit disk-quota defect (#630) and consumerless KV checkpoint defect (#631) are independent, evidenced fixes.
- This does not claim all model families hold no transient RAM. Active KV, recurrent state, speculative verification and bounded asynchronous write ownership still need RAM. Remaining cache-enabled long-context retention and process-wide write budget accounting are tracked separately.
