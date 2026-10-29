#!/usr/bin/env python3
"""MiniMax-M3.1 vision fixtures for the Swift port (Tests/MLXLMTests/MiniMaxM3VisionTests.swift).

Two references, both computed here, nothing hand-written:

* preprocessing: the bundle's own image processor (`image_processor.MiniMaxM3VLImageProcessor`, the class the bundle's
  `preprocessor_config.json` names; transformers fast processor, torch uint8 antialiased bicubic resize) on seeded noise
  images that take every `smart_resize` branch (round, the min-pixels ceil, the max-pixels floor, unchanged). Stores
  the RGB bytes, the processor's float32 `pixel_values` and `image_grid_thw`, and one PNG with its RGB bytes;
* the vision stack: vMLX Python's `MiniMaxM3VLVisionStack` (`vmlx_engine/models/minimax_m3/minimax_m3_vl.py`, the
  runtime that passes the graded image set) with seeded random weights at a tiny width, the bundle's tensor names and
  layouts (patch embedding float32 in the PyTorch (out, C, T, H, W) layout, everything else bfloat16), fed the
  processor's pixel values as bfloat16 (what vMLX's preprocess hands the model). Features of each image alone and of
  the two images in one call (`cu_seqlens` per image).

Run against a CLEAN export of the vMLX commit and the bundle's processor files (only the *.json / *.py processor files
are read, never weights):

    git -C <vmlx> archive <commit> vmlx_engine/models/minimax_m3/minimax_m3_vl.py | tar -x -C /tmp/py_vl
    python generate_vision_fixtures.py --vl /tmp/py_vl/vmlx_engine/models/minimax_m3/minimax_m3_vl.py \\
        --processor <bundle dir> --commit <commit> --out Tests/MLXLMTests/Fixtures/MiniMaxM3/vision
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

# (name, width, height): the smart_resize branch each takes at factor 28, max_pixels 451584, min_pixels 3136.
IMAGES = [
    ("round_203x97", 203, 97),  # round_by_factor on both sides (84 x 196)
    ("min_50x40", 50, 40),  # rounds to 28 x 56 = 1568 px < 3136: ceil branch, upscaled to 56 x 84
    ("max_1100x700", 1100, 700),  # 770,000 px > 451,584: floor branch, downscaled
    ("same_112x112", 112, 112),  # already a multiple of 28: no resize
]
TINY = {
    "hidden_size": 64, "num_attention_heads": 4, "num_hidden_layers": 2, "intermediate_size": 96,
    "patch_size": 14, "layer_norm_eps": 1e-5, "rope_theta": 10000.0, "num_channels": 3,
    "img_token_compression_config": {"spatial_merge_size": 2, "temporal_patch_size": 2},
}
PROJECTOR_HIDDEN = 32
FORMULA = {"max_1100x700"}


def formula_rgb(w: int, h: int) -> np.ndarray:
    """rgb[y, x, c] = (7x + 13y + 61c + (x*y mod 31)) mod 256 (the Swift test rebuilds it)."""
    y, x, c = np.meshgrid(np.arange(h), np.arange(w), np.arange(3), indexing="ij")
    return ((7 * x + 13 * y + 61 * c + (x * y) % 31) % 256).astype(np.uint8)


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vl", required=True, help="minimax_m3_vl.py from a clean export of the vMLX commit")
    ap.add_argument("--processor", required=True, help="bundle directory holding image_processor.py")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    import torch
    from PIL import Image

    sys.path.insert(0, a.processor)
    from image_processor import MiniMaxM3VLImageProcessor  # the bundle's processor class

    pre_cfg = json.loads((Path(a.processor) / "preprocessor_config.json").read_text())
    proc = MiniMaxM3VLImageProcessor(**{k: v for k, v in pre_cfg.items() if k in ("image_mean", "image_std", "patch_size")})

    rng = np.random.default_rng(20261029)
    tensors: dict[str, np.ndarray] = {}
    meta: dict = {"commit": a.commit, "images": [], "tiny_config": TINY, "projector_hidden_size": PROJECTOR_HIDDEN}
    pixel_bf16 = {}
    for name, w, h in IMAGES:
        if name in FORMULA:
            rgb = formula_rgb(w, h)
        else:
            rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            # Smooth one channel a little so the resize sees structure, not only noise.
            rgb[..., 1] = ((rgb[..., 1].astype(np.int32) + np.roll(rgb[..., 1], 1, axis=1)) // 2).astype(np.uint8)
        res = proc(images=[Image.fromarray(rgb, "RGB")], return_tensors="np")
        pv = np.asarray(res["pixel_values"], dtype=np.float32)
        grid = np.asarray(res["image_grid_thw"], dtype=np.int32)
        if name not in FORMULA:  # the large image is regenerated from its formula; only its digests are stored
            tensors[f"{name}.rgb"] = rgb
            tensors[f"{name}.pixel_values"] = pv
        bf = torch.from_numpy(pv).to(torch.bfloat16).view(torch.uint16).numpy()
        pixel_bf16[name] = bf
        meta["images"].append({"name": name, "width": w, "height": h, "grid": grid[0].tolist(),
                               "pixel_values_sha256": sha(pv), "pixel_values_bf16_sha256": sha(bf)})

    # One PNG: decode check (the stored RGB bytes are the PNG's).
    png_rgb = rng.integers(0, 256, size=(45, 61, 3), dtype=np.uint8)
    Image.fromarray(png_rgb, "RGB").save(out / "decode_61x45.png")
    tensors["decode_61x45.rgb"] = png_rgb

    # The vision stack at a tiny width, seeded weights in the bundle's names and layouts.
    spec = importlib.util.spec_from_file_location("minimax_m3_vl", a.vl)
    vl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vl)
    stack = vl.MiniMaxM3VLVisionStack({"vision_config": TINY, "projector_hidden_size": PROJECTOR_HIDDEN})
    weights = {}
    key = mx.random.key(29)
    from mlx.utils import tree_flatten

    from mlx.utils import tree_unflatten

    pairs = []
    for name, value in sorted(tree_flatten(stack.parameters())):
        key, sub = mx.random.split(key)
        r = mx.random.normal(value.shape, key=sub)
        if name.endswith("patch_embedding.weight"):
            v = (r * 0.05).astype(mx.float32)  # MLX (out, T, H, W, C); the bundle stores PyTorch (out, C, T, H, W)
            weights[name] = mx.array(np.array(v).transpose(0, 4, 1, 2, 3).copy())
        else:
            if "norm" in name and name.endswith("weight"):
                v = (1.0 + 0.1 * r).astype(mx.bfloat16)
            elif name.endswith("bias"):
                v = (0.02 * r).astype(mx.bfloat16)
            else:
                v = (r * (1.0 / value.shape[-1] ** 0.5)).astype(mx.bfloat16)
            weights[name] = v
        pairs.append((name, v))
    stack.update(tree_unflatten(pairs))
    mx.eval(stack.parameters())

    def run(names):
        pv = mx.concatenate([mx.view(mx.array(pixel_bf16[n]), mx.bfloat16) for n in names], axis=0)
        grid = mx.array([m["grid"] for m in meta["images"] if m["name"] in names], dtype=mx.int32)
        f = stack(pv, grid)
        mx.eval(f)
        return np.array(f.astype(mx.float32)), str(f.dtype)

    feats = {}
    for n in ("round_203x97", "same_112x112"):
        f, dt = run([n])
        tensors[f"features.{n}"] = f
        feats[n] = {"rows": int(f.shape[0]), "dtype": dt, "sha256_f32": sha(f)}
    f, dt = run(["round_203x97", "same_112x112"])
    tensors["features.joint"] = f
    feats["joint"] = {"rows": int(f.shape[0]), "dtype": dt, "sha256_f32": sha(f)}
    meta["features"] = feats
    meta["weights"] = {k: str(v.dtype) for k, v in sorted(weights.items())}

    # The chat template's renders of image conversations, through the bundle's HF tokenizer, in the content shape vMLX's
    # preprocessing builds (`_normalize_messages_for_template`: the image items first, then the message's text).
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.processor)
    img = {"type": "image"}
    convs = {
        "one_image": [{"role": "user", "content": [img, {"type": "text", "text": "What is in this image?"}]}],
        "two_images": [{"role": "user", "content": [img, img, {"type": "text", "text": "Which has more circles?"}]}],
        "follow_up": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": [img, {"type": "text", "text": "Describe it."}]},
            {"role": "assistant", "content": "A red square."},
            {"role": "user", "content": "What color was it?"},
        ],
        "image_only": [{"role": "user", "content": [img, {"type": "text", "text": ""}]}],
    }
    meta["renders"] = {name: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for name, m in convs.items()}
    mx.save_safetensors(str(out / "vision-fixtures.safetensors"), {k: mx.array(v) for k, v in tensors.items()})
    mx.save_safetensors(str(out / "vision-tiny-weights.safetensors"), weights)
    (out / "vision-fixtures.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"images": [(m["name"], m["grid"]) for m in meta["images"]], "features": feats}, indent=1))



if __name__ == "__main__":
    main()
