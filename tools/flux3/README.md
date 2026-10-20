# FLUX 3 local qualification

The staged 4-bit bundle is already local. These commands use the production
Swift pipeline and pinned local mflux reference; they never contact Osaurus.
Use a Metal-capable session with resolved Swift dependencies. Run each command
in its own process so the model and GPU allocations are freed between rows.
The XCTest is skipped unless both runtime environment variables are supplied.

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export FLUX3_RUNTIME_BUNDLE=$BUNDLES/FLUX-3-Image-Klein-mflux-4bit
export FLUX3_RUNTIME_OUT=$OUT/swift-gen
export FLUX3_RUNTIME_SEED=42
export FLUX3_RUNTIME_STEPS=4
export FLUX3_RUNTIME_WIDTH=512
export FLUX3_RUNTIME_HEIGHT=512
swift test --disable-sandbox --filter Flux3KleinRuntimeTests/testNativeParity

$MFLUX_PYTHON tools/flux3/reference-parity.py \
  --out $OUT/mflux-gen \
  --width 512 --height 512 --steps 4 --seed 42
$MFLUX_PYTHON tools/flux3/reference-parity.py \
  --swift "$FLUX3_RUNTIME_OUT" \
  --out $OUT/mflux-gen
```

Both defaults use the exact prompt:
`A red ceramic teapot on a wooden table, soft morning window light, detailed photograph`.
For an edit use the same prompt and reference paths in both backends:

```sh
export FLUX3_RUNTIME_REFERENCES=$OUT/mflux-gen/mflux.png
export FLUX3_RUNTIME_PROMPT='A blue ceramic teapot on a wooden table, soft morning window light, detailed photograph'
export FLUX3_RUNTIME_OUT=$OUT/swift-edit
unset FLUX3_RUNTIME_WIDTH FLUX3_RUNTIME_HEIGHT
swift test --disable-sandbox --filter Flux3KleinRuntimeTests/testNativeParity
$MFLUX_PYTHON tools/flux3/reference-parity.py \
  --out $OUT/mflux-edit \
  --prompt "$FLUX3_RUNTIME_PROMPT" --reference "$FLUX3_RUNTIME_REFERENCES"
$MFLUX_PYTHON tools/flux3/reference-parity.py \
  --swift "$FLUX3_RUNTIME_OUT" \
  --out $OUT/mflux-edit
```

For multiple references set `FLUX3_RUNTIME_REFERENCES` to newline-separated
ordered paths, and repeat `--reference path` in that same order for mflux.
Omitted edit dimensions use the first prepared reference. The result JSON
records effective output dimensions after crop-back, seed, load/operation
seconds and images/sec. Runtime logs also record denoising steps/sec.

Each backend writes `noise`, `text`, optional `references`, `step0-velocity`,
and final `latents` safetensors. The comparator reads F32/BF16 without MLX,
checks the prompt/seed/steps/dimensions/reference identity, then records tensor
and RGB8 mean/max absolute differences. It imposes no invented parity cutoff.
Inspect the PNGs and compare stage tensors before diagnosing any mismatch.
The flat/finite-image XCTest checks are diagnostic, not visual quality proof.

Pure checks (no GPU):

```sh
FLUX3_STAGED_BUNDLES=$BUNDLES python3 tools/flux3/test-contracts.py
```

Quick native check without XCTest (one process per row):

```sh
swift build -c release --product vmlxflux-probe
.build/out/Products/Release/vmlxflux-probe flux3-check $BUNDLES/FLUX-3-Image-Klein-mflux-4bit $OUT generate
.build/out/Products/Release/vmlxflux-probe flux3-check $BUNDLES/FLUX-3-Image-Klein-mflux-4bit $OUT edit <reference.png>
.build/out/Products/Release/vmlxflux-probe flux3-check $BUNDLES/FLUX-3-Image-Klein-mflux-4bit $OUT unsized <reference.png>
.build/out/Products/Release/vmlxflux-probe flux3-check $BUNDLES/FLUX-3-Image-Klein-mflux-4bit $OUT multi <ref1.png> <ref2.png>
```
