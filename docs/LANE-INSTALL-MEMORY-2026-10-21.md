# Bounded lane-weight conversion memory

Qwen27B DFlash2 app checks produced coherent connected output, but process peak reached39.83–40.34decimalGB during load/warm-up and settled to27.86GB. Actual Activity Monitor showed25.61GB after the reply. Base weights22.68GB and the bundled drafter~3.6GiB explain steady compute storage, not the transient peak.

Lane installation evaluated tiles in2GiB batches while retaining the complete original leaf-module array and every replacement until the final model update. The candidate applies each evaluated batch immediately and clears consumed leaf references. Packed scale copies, tile layout, raw-reader exclusions, fused compute and adaptive controllers remain unchanged. The existing batch ceiling remains2GiB; one matrix can exceed that ceiling and the current iteration can retain one original module until its scope ends.

A focused GPU regression uses eight2MiB packed matrices and a4MiB evaluation batch. Before the lifetime fix:17MiB additional active peak, failed resource assertion. After:3MiB, with exact output comparisons passing. All5lane classification/layout/allocation tests pass. No sampler or seed changes.

Source/test receipts: `~/vmlx-private-evidence/resident-weight-ownership-20261021/lane-install-peak-before.log` and `lane-install-peak-after-final.log`. One intermediate build failed because the test resource helper had placed generated metallibs outside the signed resource directory; removing only those generated duplicates restored normal signing. It is not a product signing workaround.

Actual Release-app comparison now confirms a lower transient process peak:33.0–33.2decimalGB versus39.8–40.3GB before, settling near27.5GB in both builds. Connected code/tool/SSD answers complete normally. Activity Monitor after the candidate reply shows25.57GB. App8d762579 pins runtimec02c82b8e; binary SHA256 c6af74d56bbe29bfda0c9550ca428fd70ef5bbd1814938c591492774fcc4c30e. Receipts: turns/Q03-bounded-dense-auto andQ04-bounded-dense-followup in the same evidence root.

Performance preservation is not cleared: reported code rate75.18→62.92tok/s and follow-up64.68/72.41→44.52/48.82 in these unmatched runs. Full prompts, output lengths, acceptance and selected adaptive widths differ; candidate GPU clock median was higher, not lower. Keep the PR draft pending controlled exact-context isolation; do not claim a speedup or dismiss the observed decline. The stronger allocation test also verifies every projection remains tiled/accelerated after partial updates. No all-family,128k,aggregate KV/writer memory or universal adaptive-speed claim follows from the focused test. No release/tag.


## Additional controls and disposition

The real chat sampler is TopPSampler; ArgMaxSampler belongs to load warm-up. Normal UI regeneration preserves the original injected user context. On the same saved 5,551-token prompt, adaptive DFlash2 runs at60.75tok/s cold (medianGPU1095.5MHz) and92.37tok/s with5550tokens restored (median1354.5MHz); normal Off runs at14.01tok/s with the same prefix restored. All complete normally. No seed or sampler was altered, so outputs/acceptance vary and rates are not fixed-output comparisons. The earlier unmatched lower rates are retained above, not hidden.

The initialization-only lifetime fix preserves every tested accelerated projection and demonstrates lower process peaks in the actual app. Engine#640 merged asb8da74a20e2089f86edba9956d3abd3479e70623 for that bounded scope. It makes no universal adaptive-speed guarantee. Wider controller/quality/128k qualification remains separate. The prior draft-hold note described the earlier evidence checkpoint.
