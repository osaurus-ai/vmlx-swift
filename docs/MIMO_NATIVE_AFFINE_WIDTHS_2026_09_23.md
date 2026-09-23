# MiMo native affine-width loader follow-up

The installed MiMo V2.6 JANG_2L iteration 2 contains native affine 3-bit/group128 experts. The merged runtime rejected them before generation because MixedQuantizedExpertCatalog accepted only 2/4/8-bit affine tensors. Layer 11 gate has U32 shape [256,2048,384] and BF16 scales/biases [256,2048,32], representing 4096 input features at 3 bits.

The catalog now accepts MLX's native affine widths 2/3/4/5/6/8. Existing payload, shape, group-size, dtype and companion validation is retained. This does not requantize weights, change dispatch defaults or widen specialized fused-kernel guards. No sampler/template/cache/memory-policy change.

## Bounded runtime proof

Private receipts are in ~/vmlx-private-evidence/mimo26-swift-2026-09-22/.

- Red: odd-affine-widths-red-r2.log reproduced InvalidBundle for 3/5/6-bit tensors on the old catalog. The earlier red.log failed scheme lookup only and is not test proof.
- Native packed bits preserved; mapped and resident single-token output matches independent native QMV exactly for 3/5/6 bits. Eight-token prefill matches native batched QMM (mapped) and SwitchGLU (resident) exactly with reordered routes. No numerical tolerance introduced.
- First prefill reference used independent single-token QMV, whose reduction shape differs from batched execution; it failed with maxAbs 1.57e-5 to 3.05e-5. Those failures remain in odd-affine-widths-green-r1.log. The native batch reference correction passed in native-affine-widths-r2.log: 12 tests.
- Full catalog suite with all three optional MiMo fused/paired flags enabled passed 13 tests in native-affine-widths-r3-optins.log. Unsupported widths 1/7/9 are still rejected; non-power-of-two gates decline specialized gate/up kernels and retain native fallback. Existing packed-bank lifetime, invalid companion/path/truncation, production-shape kernel and indexed-model/ring-cache regressions also passed.
- Final default-flags suite also passed 13 tests: native-affine-widths-r4-defaults.log and its receipt.
- All runs use TEST_RUNNER_MLX_ENABLE_TF32=0. These are actual small Metal tensor/kernel regressions, not whole-model generation; token/s is not applicable.

## Model identity and remaining app proof

Eric explicitly selected the updated installed bundle for qualification. Its actual SHA256-MANIFEST.json hash is 9c0f05f02fb4951123e3f9ff39b144071d2fb0ca84bd593bf70c879032a824fa. All 54 files, 110792864362 bytes, passed SHA256/size checks and remained stable: r21-bundle-verification.json. Old R19 model scores do not qualify this iteration.

Osaurus must consume the merged follow-up SHA, rebuild and complete the updated-bundle app/eval proof. Its draft PR #2863 is not merged by this engine change. Existing owner engine-CI waiver does not waive Osaurus CI or live app proof. Never create a release/tag or dispatch a release workflow.
