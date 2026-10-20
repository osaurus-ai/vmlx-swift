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
- Multimodal turns, final-pin SSD companion restore, controlled performance,
  and MTP On/Off app qualification remain pending.
- Python/Electron receipts for these bundles do not establish Swift app behavior.

No speculative speedup or merge readiness is claimed. Existing MTP tuning and
explicit-user activation policy remains unchanged.
