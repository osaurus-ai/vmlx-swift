# MiniMax-M3 fixtures

## `janght-reference.safetensors` / `.json` (MiniMaxM3JANGHTTests)

These are the vMLX Python runtime's logits for the `m31-janght/` bundle. `MiniMaxM3JANGHTTests` checks Swift against
them at `strictTolerance` (1e-3) with `MLX_ENABLE_TF32=0`. Swift and Python must run the same MLX quantized-matmul
kernels for that to hold.

- **MLX:** the goldens come from MLX `869ff95b0`, the commit `Source/Cmlx/mlx` points at, built as a Python wheel.
  Its version string is `0.32.3.dev20261031+869ff95b0`, and the `mlx` field of the json records it.
  - That commit includes upstream `f824bce27`, "[Metal] Add staged affine_qmm_t_nax kernels with split-k" (#4682).
  - The wheel's metallib has 84 `affine_qmm_t_nax_splitk_*` kernels. The MLX 0.32.3 wheel has none.
  - On M5 (NAX), `use_qmm_nax_splitk` routes affine bf16 / f16 matmuls with small output grids to `qmm_t_nax`. That
    kernel accumulates all of K in fp32. Before, those matmuls took `affine_qmm_t_splitk`, which rounds each K
    partition to bf16.
  - The previous goldens (MLX 0.32.3) match only a `Source/Cmlx/mlx` without that route.
- **vMLX:** `885355f0` (jjang-ai/vmlx), run from a clean `git archive` export.
- **Other packages:** mlx-lm 0.31.3, numpy 2.2.6, Python 3.13.13.
- **Bundle:** `m31-janght/` is unchanged. The generator rewrites it byte for byte.

Command (M5 Max, from an empty output directory):

    MLX_ENABLE_TF32=0 PYTHONPATH=<vmlx 885355f0 export> <python with that mlx wheel> \
        scripts/minimax_m3_fixtures/generate_janght_fixtures.py --vmlx <vmlx 885355f0 export> \
        --vmlx-commit 885355f0 --output <empty directory>

Then copy `janght-reference.safetensors` and `janght-reference.json` here.

Check: the same command on the MLX 0.32.3 wheel reproduces the previous goldens byte for byte, and two runs on the
`869ff95b0` wheel give identical files.
