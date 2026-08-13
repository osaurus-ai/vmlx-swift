#!/usr/bin/env python3
"""Write the Audex release card and copy exact source-family license files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def find_release_row(manifest: dict, repo_id: str) -> tuple[dict, dict]:
    for family in manifest["families"]:
        for bundle in family["bundles"]:
            if bundle["repo_id"] == repo_id:
                return family, bundle
    raise SystemExit(f"repo_id is not present in release manifest: {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=Path("docs/AUDEX-FAMILY-QUANT-RELEASE-2026-07-20.json"),
    )
    parser.add_argument("--license-file", type=Path, required=True)
    parser.add_argument("--license-directory", type=Path, required=True)
    args = parser.parse_args()

    bundle_path = args.bundle.resolve()
    manifest = json.loads(args.release_manifest.read_text())
    family, release = find_release_row(manifest, args.repo_id)
    conversion = json.loads((bundle_path / "vmlx_conversion.json").read_text())
    bits = release["bits"]
    size_gib = release["weight_bytes"] / 2**30
    is_hybrid = family["model_type"] == "nemotron_h_audex"
    expert_line = (
        f"- Stacked routed-expert groups: {release['stacked_expert_group_count']}\n"
        if is_hybrid
        else ""
    )
    cache_line = (
        "23 Mamba states plus 6 attention KV caches; the 23 MoE blocks own no KV state"
        if is_hybrid
        else "28 standard causal KV caches"
    )
    config_note = (
        "The source `time_step_limit: [0.0, Infinity]` field is omitted because "
        "bare `Infinity` is not strict JSON. The native Nemotron-H configuration "
        "applies the equivalent `[0.001, +infinity]` runtime default.\n"
        if is_hybrid
        else ""
    )
    turn_rates = " / ".join(str(value) for value in release["turn_tokens_per_second"])
    repo_name = args.repo_id.split("/", 1)[1]
    card = f"""---
license: other
license_name: nvidia-oneway-noncommercial-license
license_link: https://huggingface.co/{family['source_model']}/blob/{family['source_revision']}/LICENSE
base_model: {family['source_model']}
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - vmlx-swift
  - audio-language-modeling
  - audio-understanding
  - speech-recognition
  - speech-translation
  - nemotron-labs-audex
language:
  - en
---

# {repo_name}

MLX affine {bits}-bit, group-size-64 conversion of
[`{family['source_model']}`](https://huggingface.co/{family['source_model']})
at exact revision `{family['source_revision']}`.

Status: **PASS for the explicitly selected instruct audio-input/text-output
path; PARTIAL for the complete upstream feature set.** The language decoder is
quantized. The NV-Whisper audio encoder and audio projector stay in source
precision.

Runtime pull requests:

- [vMLX native Audex runtime](https://github.com/osaurus-ai/vmlx-swift/pull/168)
- [Osaurus audio-only capability and exact runtime pin](https://github.com/osaurus-ai/osaurus/pull/2116)

Until those changes merge into a release, use the PR revisions rather than an
older packaged vMLX/Osaurus build.

## Proven local row

The exact directory uploaded to this repository passed both static bundle
validation and live Apple-silicon generation on 2026-07-20.

- Model class: `{family['model_type']}`
- Weight shards: {release['weight_shards']}
- Weight bytes: {release['weight_bytes']} ({size_gib:.3f} GiB)
- Indexed tensors: {release['tensor_count']}
- Quantized modules: {release['quantized_module_count']}
{expert_line}- Source-precision audio tensors: {release['audio_tensor_count']}
- Load time: {release['load_seconds']:.1f} seconds
- Three turn rates: {turn_rates} tok/s
- Current sampled physical footprint: {release['current_phys_footprint_gib']:.3f} GiB
- Peak sampled physical footprint: {release['peak_sampled_phys_footprint_gib']:.3f} GiB
- Stop result: normal stop on all three turns
- Visible output: coherent on all three turns
- Reasoning/control-marker leakage: none in the passing instruct row

The live input was a real speech WAV. Turn 1 transcribed it. Turn 2 correctly
named vegetables from the transcript. Turn 3 correctly confirmed that mutton
was mentioned. The 96-token setting was a safety maximum; a length stop was a
failure condition and did not occur.

The sampled footprint values are direct telemetry, not a low-RAM claim.

## Required template mode

NVIDIA's template supports thinking and instruct modes. The passing runtime row
explicitly selected instruct mode at request time:

```swift
let input = UserInput(
    chat: [
        .user(
            "Transcribe the speech accurately.",
            audios: [.url(audioURL)])
    ],
    additionalContext: ["enable_thinking": false]
)
```

This is the source template's own `<think></think>` instruct path. It is not a
modified model default, forced close token, prompt rewrite, or sampler guard.

Important negative result: with `enable_thinking` omitted, representative 2B
and 30B rows stopped after emitting the transcription inside an unclosed
reasoning channel. The production stream reported 195-196 reasoning characters
and zero visible characters. Default-thinking audio is therefore not claimed
as passing. Hosts should expose the instruct choice explicitly and keep that
thinking-mode row marked partial.

## Generation parameters

The proof used the bundle generation defaults:

```text
temperature = 0.6
top_p = 1.0
top_k = 0
min_p = 0.0
repetition_penalty = nil
```

NVIDIA separately recommends greedy decoding for ASR/translation and
temperature 0.7 with top-p 0.9 for audio understanding. Those are explicit
task recipes; a host must not silently install them as family-wide defaults.

## Download

```bash
hf download {args.repo_id} \\
  --local-dir ~/models/{args.repo_id}
```

## Static verification

From the vMLX PR checkout:

```bash
python3 scripts/verify-audex-bundle.py \\
  ~/models/{args.repo_id} \\
  --expect-bits {bits} \\
  --source /path/to/{family['source_model'].split('/', 1)[1]}/checkpoint_folder_full
```

The verifier checks every safetensors header against the index, tensor
uniqueness and shard ownership, exact byte totals, complete affine triplets,
incomplete files, and the Nemotron-H stacked expert layout when applicable.
With `--source`, it also compares every audio tensor's dtype, shape, and raw
data SHA-256 to the exact source snapshot; that stronger mode passed before
this repository was uploaded.

## Architecture and cache ownership

- Cache topology: {cache_line}.
- Audio: 32-layer NV-Whisper/Qwen2 encoder, 128 mel bins, 16 kHz input.
- Audio embeddings: 750 per 30-second clip.
- Output: text tokens only in the current native Swift runtime.

{config_note}No prefix/paged/L2-disk/TurboQuant-KV hit is claimed by this release row. A
future cache claim must include counters, coherent post-hit output, and—for the
30B family—the exact Mamba companion-state restore boundary.

## Supported and unsupported surfaces

| Surface | Status |
| --- | --- |
| Text plus audio -> text | Proven in explicit instruct mode |
| Growing audio conversation -> text | Proven in explicit instruct mode |
| Default-thinking audio -> visible text | Failed representative row |
| Image/video input | Unsupported by this wrapper |
| Batch size greater than one | Unsupported; use batch size 1 |
| Text-to-speech output | Not implemented in this Swift runtime |
| Text-to-audio output | Not implemented in this Swift runtime |
| Speech-to-speech audio output | Not implemented in this Swift runtime |

The upstream repository includes separate audio-generation checkpoints and
codecs. Their presence does not make this input-audio quant an output-audio
runtime.

## Conversion provenance

```json
{json.dumps(conversion, indent=2)}
```

## 한국어 요약

이 저장소는 `{family['source_model']}`의 MLX affine {bits}비트 양자화입니다.
언어 디코더만 양자화했고 NV-Whisper 오디오 인코더와 오디오 프로젝터는
원본 정밀도를 유지합니다. `enable_thinking=false`를 명시한 instruct 모드에서
실제 음성 전사와 3턴 대화가 통과했습니다. 기본 thinking 모드는 reasoning
채널을 닫지 않아 화면에 표시할 답변이 비어 있으므로 아직 부분 지원입니다.
현재 Swift 런타임은 오디오 입력과 텍스트 출력만 지원하며 이미지, 비디오,
TTS, 텍스트-오디오 출력은 지원한다고 주장하지 않습니다.

## License

Use is governed by the NVIDIA OneWay Noncommercial License. `LICENSE.txt` and
the source-family `license/` directory are included. Review
`license/THIRD_PARTY_NOTICES.md` before redistribution or deployment.
"""

    (bundle_path / "README.md").write_text(card)
    shutil.copy2(args.license_file, bundle_path / "LICENSE.txt")
    shutil.copytree(args.license_directory, bundle_path / "license", dirs_exist_ok=True)
    print(f"wrote release metadata for {args.repo_id}: {bundle_path}")


if __name__ == "__main__":
    main()
