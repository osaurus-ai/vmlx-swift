# GLM bounded indexed cache: draft checkpoint

Status: **NOT READY**. Component alignment and serialized continuation have evidence. The focused suite still has six numeric failures, including an uncapped control. Real GLM 128k app behavior, physical footprint, sustained prefill/decode/speculative performance, and coherent multiturn/media behavior are unproven.

## Defect and implementation

A finite `maxKVSize` previously replaced GLM's indexed cache with a plain rotating cache. Sparse attention consequently lost the indexer's historical rows and used a zero query offset after the selection threshold. The defect occurred before eviction.

The capped factory now retains an indexed cache with:

- Rotating KV storage, and absolute-position-to-physical-slot mapping for selected rows.
- Bounded raw indexer storage retaining complete original boundary pools, including their learned member positions. Committed KV has at most the configured capacity; companion rows have at most capacity plus two partial pools. Multirow attention temporarily uses the union of per-query windows, at most capacity plus chunk length minus one.
- Causal/window eligibility before selection, and selected indices mapped back into retained KV storage.
- Copy, pre-eviction trim, explicit nontrimmability after eviction, versioned state metadata and validated disk restoration.

The uncapped cache path and generation settings remain unchanged. Host position bookkeeping uses at most three ranges and device integer arithmetic, with no context-length host loop or GPU readback. Companion eviction slices old spans before concatenation to avoid copying the whole payload twice.

## Observed component checks

Passing checks cover pre-eviction parity in both MLA representations; original pooled-key preservation; exact scalar selected-position/window and physical-ring-slot oracles; factory topology; native BF16/FP16 state; copy/trim; malformed restoration; and bounded plus uncapped gather/mask controls.

The actual disk test uses `TQDiskSerializer`, a safetensors file save/load, and `restoreFromDiskArrays(requirePromptBoundary: true)`. It restores a wrapped cache at position 25 and continues four rows using matching chunk geometry: **max absolute output difference 0.0**. This is a component file round trip, not app-owned L2 cache proof.

The complete component run remains **FAIL**: five capped chunk-versus-single-row cases differ by approximately 0.000873–0.001227, against the unchanged 0.0001 threshold. The uncapped control also fails at **0.001111865**. The common arithmetic gap is unresolved; no seed changes, tolerance relaxation or disabled failing cases were used.

## Append-only diagnostic

One sparse layer, 65,536 resident rows, BF16 latent rank 512, value stub width 1, packed indexer width 257, four warmup and twenty measured appends:

| Cache | ms per append |
| --- | ---: |
| Rotating KV only | 0.203396 |
| Existing unbounded indexed | 0.491185 |
| Bounded indexed, final ring implementation | 0.500077 |

An earlier chronological-copy draft took 1.667110 ms versus 0.579696 ms for its same-run unbounded control and was replaced. The final short probe does not establish sustained speed parity, model throughput, RAM eligibility or production readiness.

## Reproduction

Use an available compatible test metallib through `VMLX_TEST_METALLIB_PATH` when required by the runner.

```sh
swift test --build-system swiftbuild --filter Glm5NextBoundedCacheTests
swift test --build-system swiftbuild --filter Glm5NextBoundedCacheTests/actualDiskRoundTrip
VMLX_GLM_CACHE_APPEND_PROBE=1 swift test --build-system swiftbuild --filter Glm5NextBoundedCacheTests/appendCostProbe
```

Keep the PR in draft for independent review. Do not merge or release until the numeric limitation is understood and real model/app 128k, footprint, sustained speed, multiturn and relevant cache/media proofs are complete.
