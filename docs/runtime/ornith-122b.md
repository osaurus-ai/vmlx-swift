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
Bundles must declare the existing supported value:

```json
{"runtime": {"norm_convention": "qwen3_5_language_mlx_plus_one"}}
```

in `jang_config.json`. This is a storage declaration, not a sampler override.
It does not change tensor contents. The original private 122B artifacts lacked
this declaration; their metadata correction requires separate qualification.
Do not bypass their Osaurus minimum-version requirement.

## Verification status

- Release engine build: passed with Xcode's developer directory selected.
- Focused admission and converter-rehearsal tests: in progress.
- Real JANG_4M Swift diagnostic with original metadata reproduces repeated,
  incoherent reasoning and no visible answer. Corrected-metadata comparison
  and Osaurus UI proof are pending.
- JANGH2 real-bundle execution, multimodal turns, SSD companion restore,
  physical footprint, and MTP On/Off: pending.
- Python/Electron receipts for these bundles do not establish Swift app behavior.

No speculative speedup or merge readiness is claimed. Existing MTP tuning and
explicit-user activation policy remains unchanged.
