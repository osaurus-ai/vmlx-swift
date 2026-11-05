# Native disk leaf-boundary regression tests

The original fixture changed `CacheList.offset` while its recurrent child
remained at the token-key boundary. The wrapper inherits an unused offset;
`CacheCoordinator.storeAfterGeneration` intentionally validates actual leaf
offsets. The old test therefore did not create the corruption it claimed.

No production change is needed. The corrected test damages a concrete
attention or recurrent leaf while keeping the wrapper label valid, with flat
and nested caches at7/9tokens (neither divides the four-token page size).
Eight mismatched-store cases require zero stores/durable/validated entries;
repairing only the leaf then requires a successful store and reopen in all
eight cases. Two opposite controls keep valid leaves with wrapper offsetzero,
then require real disk fetch/restore, correct offsets/persistent state and no
scratch-slot persistence. Existing malformed-payload, media/token identity,
occupancy, reopen, sidecar and paged-companion controls remain.

Normal macOS Xcode reproduces the original6testfunctions/1suite with3failed
assertions. Corrected7functions/1suite pass with0failures/skips; swift-format
strict lint of the whole changed test file passes. An intermediate test-only
compile failure (mutating a read-only protocol property) is retained; the
fixture now casts its concrete test leaf without changing any production API.

SOURCE EVIDENCE: `Tests/MLXLMCommonFocusedTests/NativeDiskDurabilityTests.swift`;
production Libraries/Source/Vendors/Package.swift remain byte-identical to main
`4ba8946804941a65e376e5739ebff94909326cca`.
LIVE EVIDENCE: local `goal-storage-cache-resume-20261104/full-focused-xcode/`
`native-boundary-fixture-baseline` and `native-boundary-fixture-fixed-v2`
normal Xcode commands/logs/xcresults/source hashes; actual MLX arrays, store,
file validation, reopen and restore execute production cache functions.
No new app runtime behavior: existing Release38 six-turn canonical GUI SSD/
restart proof remains applicable to the unchanged production source. This is
not a new model-family, numerical, speed or low-RAM qualification. Prior wider
suite failures remain open. No AgentLoop/app-tool changes or release/tag.
