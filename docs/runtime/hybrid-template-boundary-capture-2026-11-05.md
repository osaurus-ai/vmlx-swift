# Capture the boundary that finalization already publishes

An actual GLM image-history follow-up restored 1,060 of 1,129 prompt tokens from disk, but then replayed the first 1,127 tokens after answering. The replay took 4,220 ms of a 4,321 ms completion tail. SSD writes took 63 and 14 ms. The next optimization target is the synchronous boundary replay, not disk throughput or app orchestration.

Media processors may deliberately omit canonical history boundary counts. Finalization already derives a safe stripped boundary through `TokenIterator.hybridStripBoundaryIndex`, using the active template suffix when counts are absent. Batched prefill capture previously inspected only the explicit counts, so it missed that same target in an otherwise validated, text-only restored suffix.

Use the existing shared calculation for inline capture too. The prepared suffix must still be text-only; post-prepare key paths, auxiliary prompts, disabled storage and invalid offsets remain excluded. Cold media and changed media keep their full payload. No boundary publication, GPU drain, sampler, template, model setting or app-tool implementation is removed or altered.

A tiny cache-recorder baseline forwards the seven-token suffix and then unnecessarily forwards the eleven-token old image prefix. The new absent-count warm method fails while all three existing controls pass. The candidate passes six methods, including warm batch-one/two and cold/changed-media controls without canonical counts. Final fixture formatting passes strict lint. These deterministic dispatch/cache checks are not model quality or speed measurements.

Full focused regression and source-bound actual Osaurus GUI qualification remain required before promotion. Record real growing-turn cache offsets, companion state, visible and expanded reasoning, token rates, completion tail and physical footprint. Keep any failed states before further requests. No broad family, speedup, low-RAM or numerical-parity claim follows from the component pass. No release or tag changes.
