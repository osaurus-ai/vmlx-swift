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

## Numerical diagnosis follow-up

The first observed difference is before cache storage or sparse selection. A shared FP32 attention module with identical input and weights produced:

| Stage/control | Maximum absolute difference |
| --- | ---: |
| q_a batched projection vs single-row projection | 0.0015132427 |
| q_a batched projection vs host double-precision dot products | 0.0015133619 |
| q_a single-row projection vs the same double-precision dots | 0.00000023841858 |
| RMSNorm applied batchwise vs rowwise to identical projected inputs | 0.0 |
| Other same-input Linear stages | approximately 0.000800–0.001513 |

Source tracing supports a row-geometry arithmetic distinction: Linear calls matmul; Metal matmul selects GEMV for a singleton matrix dimension, whereas matrix GEMM can select NAX when the default-enabled TF32 policy permits FP32 inputs. This is source-path evidence plus the double-dot comparison, not a captured Metal kernel trace. No TF32 or other production precision policy was changed.

The selection threshold cannot explain all original failures: topK=4 aligns with the first four-row chunk, and topK=64 never selects across the 37-row fixture. A focused topK=8, pool-eight control evaluates early rows inside a 16-row selection call and proves their selected mask is exactly causal. The real dense/absorbed arithmetic still differs by row shape.

An additional same-module FP32 probe observed an uncapped row-25 output difference of **0.23808202** (absorbed) and **0.2380518** (expanded), alongside roughly 0.001 differences on other rows. Selection amplification near a ranking boundary is a hypothesis, not established for that outlier; it remains visible and unresolved.

The inspected GLM bundle uses BF16-backed affine-eight projections with group size 64, except indexer wq_b using MXFP8 with group size 32. The original unquantized FP32 fixture therefore does not establish actual quantized bundle behavior. An additional tiny generated-weight fixture using those formats observed projection row-shape differences of **0.0078125**, and attention differences of **0.00390625** for both uncapped and cap-seven cases. This reproduces a row-shape precision distinction in the format-matched synthetic path; it is not real-weight inference, a JANGHT routed-kernel check, or a coherence verdict.

The new stage probes report measurements without asserting parity. The original strict failing tests remain unchanged. No production arithmetic fix, changed seed, tolerance relaxation, or disabled failure was introduced. Reproduce the added bounded diagnostics with filters Glm5NextBoundedCacheTests/numericalStageProbe, /nativeQuantizedStageProbe, and /thresholdCrossingSelectionControl.

## Selection-only counterfactual

A further 257-row diagnostic compares native indexer selection from single-row calls against chunks repeating 4, 3, 9, 1, 12, 8, using one shared module and identical inputs. Both plain FP32 and synthetic BF16 affine-eight/MXFP8 variants observed **zero selected-key-set flips**. Holding all attention projections and operands fixed and changing only the observed selection masks consequently gave **maxabs 0.0**; unchanged-mask rows are asserted exactly equal.

This run does not reproduce the earlier row-25 outlier and does not establish its cause. It supplies a counterfactual diagnostic that can attribute selection-only effects when flips occur; no such effect was observed in this run. Generated weights are not seeded or tuned, so later invocations may have different ranking margins. The diagnostic can report host double-dot ranking margins for flipped rows, but that branch was not exercised in this receipt. It is not proof of chunk parity or actual model behavior. The six strict numerical failures remain unresolved and enabled.

Reproduce with `swift test --build-system swiftbuild --filter Glm5NextBoundedCacheTests/selectionAmplificationProbe`. Runtime source and precision policy are unchanged.
