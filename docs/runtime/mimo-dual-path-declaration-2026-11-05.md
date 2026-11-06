# MiMo v2 registry overlap declaration

The focused runtime registry checks report `mimo_v2` in both factories but absent from `DualPathFamilies.modelTypes`. Add that existing overlap to the declaration.

The LLM entry selects MiMoV2Flash or the V2.6 text implementation by the bundle contract. The VLM entry accepts only the V2.6 contract and otherwise throws. The declaration records registry overlap, not a promise that every MiMo v2 bundle has vision. Factory dispatch, model construction, cache, generation defaults, parser behavior and app execution are unchanged. No model bundle is modified or loaded for this correction.

Baseline evidence: the normal full focused run at runtime42e77371 reports two failed methods in `DualPathFamiliesTests`, each explicitly naming only undeclared `mimo_v2`. The production files are unchanged between that source and main2f404cdd. Existing live registry and synthetic divergence suites retain both directions, missing-half registration checks and nonempty registry guards.

Validation: the normal `xcodebuild test` run passed all nine tests in the two existing live-registry and synthetic-divergence suites, with exit zero. No tests were weakened or skipped. This correction does not qualify MiMo inference, multimodal payloads, speed or RAM, and does not close the remaining unrelated focused failures. No AgentLoop/tool implementation, release or tag changes.
