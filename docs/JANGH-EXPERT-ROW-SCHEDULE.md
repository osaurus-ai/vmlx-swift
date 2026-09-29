# JANGH expert-aligned prefill scheduling

Candidate, not runtime validated. Decode dispatch and packed-codebook math are unchanged.

The sorted prefill path computes GPU lower-bound row offsets for each expert, using the same binary-search semantics as MLX gather_mm_offsets. The scheduler partitions each expert's contiguous rows into16-row steel or64-row NAX tiles. Each threadgroup processes one expert rather than repeating the full reduction for every expert boundary in a globally aligned tile. NAX input loads must be bounded by the scheduled expert row count, including the second simdgroup; bounding only against total M could include the next expert.

One extra expert group collects every out-of-range sorted route and writes NaNs without accessing bank weights, preserving the existing invalid-route contract. Empty experts create no work. Dispatch uses the safe upper bound min(M, ceil(M/BM)+E); surplus groups return before barriers or outputs. Sorted IDs are a caller precondition, already provided by routed argSort.

Validation pending: build, both-backend existing prefill numerical suites, new empty-expert/lane-boundary/partial-tile/invalid-suffix matrix, full routed weighted-reduction tests, real model prefill and multi-turn/cache proof. Current prototype builds offsets separately for gate/up and down; sharing one offset tensor per routed call is a possible later optimization. No measured speedup claimed.

## Source safety audit and pending proof

- Lower-bound metadata is computed entirely on GPU. No CPU route readback is introduced.
- Expert-count and M guards leave room for signed32-bit scheduler ceil-division and32-lane scans. Every valid output row is assigned exactly once under the sorted-route contract; excess grid tiles return uniformly before barriers.
- Inactive NAX simdgroups still participate in shared weight loads/barriers but do not load activation fragments. Ragged expert tails bound loads/stores using scheduled rows. Weight loads retain their existing column bounds.
- Existing immutable offsets are padded to at least8 elements to keep device-pointer ABI matching custom Metal input conventions; padding is never referenced.
- Independent CPU scheduling coverage checked1540 randomized shapes/distributions, BM16/64, E1/2/31/32/33/65/256, M1 through513, including invalid suffix groups. This verifies the bounds plan, not shader behavior.
- Existing tests now cover all2/3/4/6/8 bit widths in the new scheduling edge matrix. Existing tolerances are unchanged.

Pending: compile MLXLMTests against this candidate; run JANGHPrefillKernelTests (6 methods), JANGHRoutedPrefillTests (1), and JANGHSelectedExpertTests/testDiagnosticLayerMatchesDecodeAndPrefillWithoutRetainingWholeBanks (1). Then reuse the exact candidate RunBench for matched old/new long-prompt prefill and native multi-turn/cache proof. Decode unchanged still needs full-model smoke coverage to detect accidental dispatch effects. Do not count pure CPU oracle or earlier binary tests as candidate Metal proof.
