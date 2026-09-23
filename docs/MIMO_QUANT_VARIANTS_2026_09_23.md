# MiMo quantized bundle variants

The earlier fused-QKV MiMo selector required both affine and MXFP4 metadata.
Uniform affine, uniform MXFP4 and MXFP8 variants therefore selected the legacy
text model (or failed VLM selection), despite retaining the V2.6 architecture.
The shared contract now selects the fused runtime from model_type=mimo_v2,
attention_projection_layout=fused_qkv and a nonempty quantization dictionary.
Tensor loaders continue to validate individual formats and companions.

The indexed expert catalog now distinguishes native MXFP8 from MXFP4 using
validated packed geometry: U32 weights, U8 microscaling companions, no affine
biases, group32 and respectively 8 or 4 bits. Affine retains its separate
bias/scale contract: bits2/3/4/5/6/8, group32/64/128, matching F16 or BF16
companions. Packed data and companion dtypes are preserved without conversion.
MXFP8 uses native MLX routed matmul; the affine/MXFP4-only optional kernels
retain their existing guards. No new fusion is enabled by default.

This describes native MLX packed formats, not arbitrary source FP8 checkpoint
encodings, NVFP4, or a claim that every downloadable MiMo bundle is qualified.
The current installed iteration2 has affine2/3 and MXFP4 experts; its complete
bundle qualification and audio semantic failures remain in the Osaurus proof.

## Regression design and evidence

Private receipts: ~/vmlx-private-evidence/mimo26-swift-2026-09-22/.

- native-quant-matrix-r1-defaults: existing supported formats passed. The added
  affine matrix has36 mixed-bank configurations covering every gate/up/down
  role at all six widths, all three group sizes and F16/BF16. Fourteen MXFP4
  configurations cover every nonempty role combination with both input dtypes.
- native-quant-matrix-r2-fp8-red: old source fails14 MXFP8 combinations plus
  nine architecture/admission assertions across uniform affine/MXFP4/MXFP8.
  Exit65, no host-guard trip. These failures are retained.
- Every projection's packed weight, scales and biases must retain exact bytes
  and dtype. Mapped and resident paths exercise single-token decode and
  eight-token prefill with reordered routes. Decode uses independent native
  projections; prefill compares matching native grouped QMM / SwitchGLU batch
  geometry. All numerical comparisons require exact equality.
- A tiny indexed model loads actual safetensors and compares warm prefill,
  rotating-cache wrap and subsequent decode against the native reference,
  parameterized across affine3, MXFP4 and MXFP8 gates.
- Tests use MLX_ENABLE_TF32=0 and run with optional fusions off and on. These
  bounded Metal regressions do not generate language, so token/s is N/A.

Final corrected-source proof:

- native-quant-matrix-r4-defaults-receipt.json:18 tests passed, including64
  format-matrix cases (36affine,14MXFP4,14MXFP8), malformed MX companions,
  architecture/admission checks and indexed-model ring-cache parity.
- native-quant-matrix-r5-optins-receipt.json:27 tests in two suites passed with
  all three optional MiMo fusion flags enabled, including the existing MiMo
  runtime suite and legacy/fused architecture dispatch regression.
- Both runs exited0 without a host-guard trip. Normal memory pressure, zero
  swap. No production-sized model was loaded during these checks.
- The paired Osaurus PR must pin the merged follow-up SHA, rebuild, and repeat
  affected current-bundle proof. These results do not resolve the retained
  audio semantic failures or qualify an untested full MXFP8 bundle.
No release, tag or release-workflow dispatch is authorized.
