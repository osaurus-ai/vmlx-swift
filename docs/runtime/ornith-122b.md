# Ornith 1.5 122B runtime qualification

## Bundle contract

Ornith's `qwen3_5_moe` bundles contain 48 routed layers, 256 experts per layer,
and top-8 routing. The JANGH bundle uses per-projection 2/3-bit Hadamard-32
codebooks for backbone routed experts. Vision, attention, shared experts and
its separate native MTP head remain affine or unquantized as declared.

The factory admits the actual configuration and safetensor layout, never the
repository name. Custom banks must cover every routed backbone layer; ordinary
quantization keeps its own plan, including the affine MTP experts. Text-only
requests use the same validated VLM construction without requiring vision input.

## Norm storage requirement

The AWQ exporter stores Qwen language norms as `(1 + weight) / scale - 1`.
These raw deviations require `+1` at load. AWQ can move their means above 0.5,
so the legacy mean-based fallback cannot reliably identify this storage state.
Bundles must declare the explicit raw-storage value:

```json
{"runtime": {"norm_convention": "qwen3_5_raw_weight"}}
```

in `jang_config.json`. This is a storage declaration, not a sampler override.
It does not change tensor contents. The original private 122B artifacts lacked
this declaration; their metadata correction requires separate qualification.
Do not bypass their Osaurus minimum-version requirement.

## Verification status

- Release engine build: passed with Xcode's developer directory selected.
- Focused admission and converter-rehearsal tests: 11 passed, one unrelated
  optional dense-Qwen artifact test skipped. The rehearsal executes with and
  without its affine MTP head, through the public load configuration.
- Real JANG_4M Swift diagnostic with original metadata reproduces repeated,
  incoherent reasoning and no visible answer. A metadata-only comparison on
  the same tensors completes with correct arithmetic and separate reasoning.
  Both real bundle layouts now load in optimized Swift diagnostics.
- JANGH2 Osaurus Release AR completes a connected arithmetic/tool turn with a
  visible correct answer; the first step decoded at 33.9 token/s and the cached
  tool continuation at 33.6 token/s. This run overlapped compilation, so these
  are observed rates, not a controlled performance comparison.
- Optimized inherited convenience construction previously lost the Qwen35MoE
  dynamic type and skipped its sanitizer. The designated initializer is now
  module-internal; the tiny and real MTP bundles load under optimization.
- Real JANGH2 staged verify currently differs from sequential AR logits at
  widths 2/4/8 (48 rows tested, greedy argmax unchanged). This is an open
  speculative correctness investigation, not evidence of row exactness.
- Final loader-pin Release app (engine c080bfb1a, app 2f4be285): JANGH2
  correctly reads a real image (ORCHID47 / blue square), recalls it and a
  previous label, and completes same-model Helper delegation with calculator
  use and parent continuation. Child tool continuation restores 2,444 tokens
  from SSD; parent continuation restores 6,132. First-image recall changes the
  media salt and misses, which remains an open issue.
- JANG_4M completes a 10,384-token code answer at 32.2 token/s and correctly
  recalls CEDAR-82 in a connected follow-up. Its generated code fails one of
  its own assertions; this is not an all-quality pass or proof of a runtime
  cause. A separate Orchestrator attempt was refused at child-memory admission.
- The follow-up restores only 2,613 of 13,160 tokens; the old canonical hybrid
  policy discarded the generated-answer checkpoint. Generation-submit TTFT is
  17.681 seconds, excluding a preceding 5.403-second reload. Engine #619 tracks
  a separate candidate; no fixed-app latency claim yet.
- Explicit On (Adaptive) persists in the UI, but this untuned bundle remains
  AR. MTP On/Off speed qualification and sustained 128k-context proof remain
  open; the 10k-output row is not 128k-context evidence.
- Actual SSD headers contain BF16 attention/conv and F32 recurrent state. The
  `kv=fp16` key does not describe all tensor storage. Mapped weight pages also
  make app physical footprint alone insufficient for total model-RAM claims.
- Python/Electron receipts for these bundles do not establish Swift app behavior.

No speculative speedup or merge readiness is claimed. Existing MTP tuning and
explicit-user activation policy remains unchanged.
