# MiMo V2.6 numerical fixtures

These generators use tiny, deterministic synthetic weights and inputs. They
do not load the production model. The reference is the matching JANG Python
implementation; each generator verifies its SHA256 against the committed
fixture metadata before executing it.

Use an isolated Python environment with `mlx==0.32.2` and `numpy==2.4.5`. On this Mac,
run each generator into a separate output directory, for example:

```sh
python scripts/mimo_v26_fixtures/generate_vision_fixture.py \
  --reference ../jang/jang-tools/jang_tools/mimo_v2/v26_vision.py \
  --output /tmp/mimo-v26-fixtures

python scripts/mimo_v26_fixtures/generate_audio_features_fixture.py \
  --reference ../jang/jang-tools/jang_tools/mimo_v2/v26_audio.py \
  --output /tmp/mimo-v26-fixtures
```

Run `generate_audio_encoder_fixture.py` and
`generate_audio_tokenizer_fixture.py` with the same audio reference arguments.
Compare every tensor's name, shape, dtype, and value against
`Tests/MLXLMTests/Fixtures/MiMoV26` before replacing anything. Safetensors file
hashes can differ because serialization order is not part of the numerical
contract. The portable generators reproduced all 160 tensors bit-exactly on
2026-09-22; all four safetensors hashes also matched. The committed provenance
identifies the portable scripts and preserves the initial generator identity
as `original_generator_sha256`.

The audio-tokenizer generator now explicitly uses CPU F32, matching the
production encoder/RVQ path. Its original GPU-default golden features depended
on M5 TF32. The CPU refresh preserves all weights, input mels, and RVQ codes
bit-exactly; only the two raw feature arrays change. The regression compares
those features on CPU and verifies the caller's default device is restored
after successful tokenization and errors. This fixture is valid with either
global TF32 setting.
