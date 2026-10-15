# Allosaurus on 48 GiB Macs: runtime correction in progress

Status: partial; no real 48 GiB hardware result yet.

The automatic JANGH allocator budget previously reverted to 70% of RAM if weights plus 4 GiB exceeded Metal's recommended working set. A 34.43 GiB Allosaurus bundle consequently received a 33.6 GiB limit. The allocator accounts mapped buffers and drains its reusable pool above its GC threshold. The correction clamps the automatic allowance to the smaller of the GPU ceiling and 85% physical RAM when the weights fit; explicit user limits and Strict are preserved.

`memorySafety.keepModelInMemory` is a persisted Boolean, default true, forwarded through both load resolvers. The loader applies the bounded residency target, or zero for Off/non-fitting models so the previous model's process-wide budget does not remain. The 48-to-below-64-GiB machine class retains a ten-GiB OS reserve. Smaller and larger machines keep their prior reserve. This does not change idle unload, n-gram SSD access, KV precision, samplers, kernels, or MTP acceptance/depth policy.

Native MTP now forwards completed model-prepare progress to the BatchEngine stream, translating segment-local counts after cache restores and prompt-boundary capture. No new forward calls or evaluation boundaries are introduced. Completion is emitted after evaluation of the first sampled token. RunBench records the first prefill stage independent of its detail string and prints originating failure metadata, which its previous zero-token report omitted.

## Verification so far

- 22 XCTest cases passed: nine JANGH load-policy, eleven native MTP boundary/progress, two failure-transport tests.
- Eight residency-policy tests passed, including 16-GiB safety and 48-GiB ceiling/reserve cases.
- BatchEngine MLX failure test passed both preparation/decode injected-error cases.
- MTP progress tests preserve forward lengths, input order and output tokens, including restored disk prefixes and segmented prefill.
- App setting/search/guide wiring and localization check implemented; live proof pending.

## Remaining gates and caveats

- Real M4 Pro 48 GiB recommendedMaxWorkingSetSize is unknown. Simulated pressure does not prove that ceiling.
- This allocator additionally caps GC at 95% of recommended working set; an insufficient ceiling cannot be solved by raising the Swift limit alone.
- Allosaurus proposal head adds about 0.33 GiB beyond root-shard accounting. Its actual load path and transient allowance must be included in proof.
- Need isolated Release UI toggle/persistence/reload, cold and restored multiturn, sustained prose/easy-code MTP on/off, clock/page-ins/phys_footprint/wired/SSD timing receipts.
- Need capture of the originating zero-token error under controlled pressure. Existing deferred construction transports errors; the benchmark used in the report omitted that field. Do not claim the original pressure failure fixed from injected-error tests.
- No release/tag or all-device performance claim.
