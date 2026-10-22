# Bounded lane-weight conversion memory

Qwen27B DFlash2 app checks produced coherent connected output, but process peak reached39.83–40.34decimalGB during load/warm-up and settled to27.86GB. Actual Activity Monitor showed25.61GB after the reply. Base weights22.68GB and the bundled drafter~3.6GiB explain steady compute storage, not the transient peak.

Lane installation evaluated tiles in2GiB batches while retaining the complete original leaf-module array and every replacement until the final model update. The candidate applies each evaluated batch immediately and clears consumed leaf references. Packed scale copies, tile layout, raw-reader exclusions, fused compute and adaptive controllers remain unchanged. The existing batch ceiling remains2GiB; one matrix can exceed that ceiling and the current iteration can retain one original module until its scope ends.

A focused GPU regression uses eight2MiB packed matrices and a4MiB evaluation batch. Before the lifetime fix:17MiB additional active peak, failed resource assertion. After:3MiB, with exact output comparisons passing. All5lane classification/layout/allocation tests pass. No sampler or seed changes.

Source/test receipts: `~/vmlx-private-evidence/resident-weight-ownership-20261021/lane-install-peak-before.log` and `lane-install-peak-after-final.log`. One intermediate build failed because the test resource helper had placed generated metallibs outside the signed resource directory; removing only those generated duplicates restored normal signing. It is not a product signing workaround.

Actual Release-app peak/output/speed comparison of this candidate remains pending. No all-family,128k,aggregate KV/writer memory or universal adaptive-speed claim follows from the focused test. No release/tag.
