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

## TF32 two-arm control

The chunk-versus-single-row gap was re-run with the one supported precision switch in the test process only: `MLX_ENABLE_TF32` (`mlx/utils.h` `env::enable_tf32`, default on; consumed by the `use_nax` predicate in `backend/metal/matmul.cpp`, `quantized.cpp` and `scaled_dot_product_attention.cpp`). Nothing in the product, the tests, the tolerances, the seeds or the weights was changed between arms; the variable is the GEMM precision policy of the float32 projections.

| Case (1e-4 tolerance) | Default (TF32 on), 5 runs | `MLX_ENABLE_TF32=0`, 4 runs |
|---|---|---|
| capped (7,4,4) | 0.00091 / 0.00112 / 0.00116 / 0.00101 / 0.00130 FAIL | pass |
| capped (9,4,4) | 0.00081 / 0.00094 / 0.00109 / 0.00092 / 0.00106 FAIL | pass |
| capped (3,4,4) | 0.00111 / 0.00093 / 0.00106 / 0.00115 / 0.00121 FAIL | pass |
| capped (7,64,4) | 0.00084 / 0.00097 / 0.00133 / 0.00099 / 0.00077 FAIL | pass |
| capped (7,8,8) | 0.00083 / 0.00074 / 0.00104 / 0.00103 / 0.00101 FAIL | pass |
| uncapped control | 0.00101 / 0.00085 / 0.00103 / 0.00096 / **0.262** FAIL | 2.4e-7 / 2.4e-7 / 2.2e-7 / 3.0e-7 pass |
| q_a batch vs double-dot oracle | 0.00135 | 4.8e-7 |
| attention cap=nil vs cap=7 (stage probe) | 0.00081–0.0017, same magnitude | 2.4e-7–2.7e-7 |

With TF32 off every strict test passes, the uncapped control and the capped cases alike, and the chunked projections agree with a host double-precision oracle to 5e-7. The 1e-3 chunk gap is therefore the multi-row TF32 NAX GEMM versus the full-precision singleton GEMV, present identically on the existing uncapped path; it is not introduced by the bounded cache. The BF16 affine8/MXFP8 stage differences (0.0078 / 0.0039) are unchanged between arms, as expected for non-float32 inputs.

The ~0.24 outlier class reproduced. The MLX default random key is time-seeded (`mlx/random.h` `KeySequence::default_`) and the fixtures do not seed it, so the historical row-25 fixture is unrecoverable; the property is not. Under the default arm, three of five processes produced one ~0.18–0.27 row: twice in the selection probe (rows 42 and 232), where the chunked and single-row paths selected different pools and the host double-dot ranking showed the top two pool scores separated by 1.5e-5 and 1.4e-8 respectively, and once in the strict **uncapped** control itself (0.262). With TF32 off, zero flips and no outliers in four processes. The outlier is a near-tie top-k pool flip amplified from TF32 rounding of the indexer scores on the existing uncapped path; it is not caused by the cap.

A precision-matched contract was added as an additional test, `chunkingPrecisionMatched`: the identical chunk plan, caps, pool sizes and 1e-4 tolerance, executed on the CPU stream where float32 matmul has a single precision for every row count. All six cases, including the uncapped control, pass at 0.9e-7–3.0e-7 under both environment arms. The original six strict Metal tests remain unchanged and still fail under the default policy; they are retained as the record that the default-precision Metal path is not row-shape invariant.

Not shown here: actual GLM bundle inference, 128k context, app behaviour, sustained speed, and whether near-tie indexer flips under default TF32 affect real-model output. No production precision policy was changed.
