# JANGH expert-aligned prefill scheduling

Candidate, not runtime validated. Decode dispatch and packed-codebook math are unchanged.

The sorted prefill path computes GPU lower-bound row offsets for each expert, using the same binary-search semantics as MLX gather_mm_offsets. The scheduler partitions each expert's contiguous rows into16-row steel or64-row NAX tiles. Each threadgroup processes one expert rather than repeating the full reduction for every expert boundary in a globally aligned tile. NAX input loads must be bounded by the scheduled expert row count, including the second simdgroup; bounding only against total M could include the next expert.

One extra expert group collects every out-of-range sorted route and writes NaNs without accessing bank weights, preserving the existing invalid-route contract. Empty experts create no work. Dispatch uses the safe upper bound min(M, ceil(M/BM)+E); surplus groups return before barriers or outputs. Sorted IDs are a caller precondition, already provided by routed argSort.

Validation pending: build, both-backend existing prefill numerical suites, new empty-expert/lane-boundary/partial-tile/invalid-suffix matrix, full routed weighted-reduction tests, real model prefill and multi-turn/cache proof. Current prototype builds offsets separately for gate/up and down; sharing one offset tensor per routed call is a possible later optimization. No measured speedup claimed.
