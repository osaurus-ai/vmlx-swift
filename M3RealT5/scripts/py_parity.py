"""Python vMLX side of the Swift-vs-Python MiniMax-M3.1 parity on the 5-layer real-weight bundles (test harness).

Same protocol as the Swift harness (M3RealT5 parity): the token sequence of tokens.safetensors, prefill in 2048-token
chunks, then teacher-forced decode steps. Per run: lp_N = log-probs (float32) of the last chunk's rows + the decode
rows; am_N = argmax and tlp_N = log-prob of the next token at every position.

Modes (one process, model loaded once; MSA env read per call):
  dense      VMLX_MINIMAX_MSA=dense (dense mask path, score-sheet selection)
  selref     auto attention, VMLX_MINIMAX_MSA_SELECT=reference (R7-equivalent selection)
  auto       auto attention, R8 selection kernels
  perturb    VMLX_MINIMAX_MSA=sparse with a perturbed selection: block scores of the 16th and 17th best block of every
             (row, group) swapped (control: must not match)
  shift      dense, input shifted by one token (control)

  MLX_ENABLE_TF32=0 python py_parity.py --bundle B --vmlx PKG --tokens tokens.safetensors --out DIR --tag T \
      --modes dense,selref [--lens 8269,32845] [--decode 16] [--sort 512]
  python py_parity.py --make-tokens --corpus C --tokens tokens.safetensors --need 32862
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def make_tokens(corpus, out, need):
    import numpy as np
    from safetensors.numpy import save_file
    rows = sorted((json.loads(l) for l in open(corpus)), key=lambda r: r["id"])
    seq = []
    for r in rows:
        if r["set"] == "ref":
            seq += r["ids"]
        if len(seq) >= need:
            break
    assert len(seq) >= need, f"only {len(seq)} tokens"
    save_file({"tokens": np.asarray(seq, dtype=np.int32)}, out)
    print("tokens", len(seq), "->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-tokens", action="store_true")
    ap.add_argument("--corpus"); ap.add_argument("--need", type=int, default=40000)
    ap.add_argument("--bundle"); ap.add_argument("--vmlx"); ap.add_argument("--tokens", required=True)
    ap.add_argument("--out"); ap.add_argument("--tag", default="t5")
    ap.add_argument("--modes", default="dense,selref")
    ap.add_argument("--lens", default="8269,32845"); ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=2048); ap.add_argument("--sort", type=int, default=512)
    a = ap.parse_args()
    if a.make_tokens:
        return make_tokens(a.corpus, a.tokens, a.need)
    assert os.environ.get("MLX_ENABLE_TF32") == "0", "run with MLX_ENABLE_TF32=0"
    sys.path.insert(0, str(Path(a.vmlx).expanduser()))
    import mlx.core as mx
    from vmlx_engine.utils import jang_loader
    import vmlx_engine.jangt.switch as SW

    default_sort = SW.SORT_THRESHOLD
    SW.SORT_THRESHOLD = a.sort
    toks = mx.load(a.tokens)["tokens"].astype(mx.int32).tolist()
    t0 = time.time()
    model, _ = jang_loader.load_jang_model(str(Path(a.bundle).expanduser()))
    print(f"loaded in {time.time() - t0:.1f}s; sort threshold {default_sort} -> {SW.SORT_THRESHOLD}", flush=True)
    lens = [int(x) for x in a.lens.split(",")]
    # The model's own msa module (jang_loader may load minimax_m3.py standalone, so `import msa` there is not
    # vmlx_engine.models.minimax_m3.msa): patch the object the attention layers call.
    msa = sys.modules[type(model.model.layers[0].self_attn).__module__].msa
    print("model msa module:", msa.__name__, msa.__file__, flush=True)

    orig_choose = msa.choose_path
    picks = []

    def spy_choose(sel, q, k, mask, n_kv):
        r = orig_choose(sel, q, k, mask, n_kv)
        picks.append(f"{r}:{int(q.shape[2])}x{int(sel.sk)}")
        return r
    msa.choose_path = spy_choose
    trace = {}
    orig_scores = msa.block_scores
    perturbed_calls = {"n": 0}

    def perturbed_scores(idx_q, idx_k, offset, block, local):
        s = orig_scores(idx_q, idx_k, offset, block, local)
        if s.shape[-1] <= 16:
            return s
        perturbed_calls["n"] += 1
        srt = mx.sort(s, axis=-1)
        v16, v17 = srt[..., -16:-15], srt[..., -17:-16]
        return mx.where(s == v16, v17, mx.where(s == v17, v16, s))

    def set_mode(m):
        for k in ("VMLX_MINIMAX_MSA", "VMLX_MINIMAX_MSA_SELECT"):
            os.environ.pop(k, None)
        msa.block_scores = orig_scores
        if m in ("dense", "shift"):
            os.environ["VMLX_MINIMAX_MSA"] = "dense"
        elif m == "selref":
            os.environ["VMLX_MINIMAX_MSA_SELECT"] = "reference"
        elif m == "perturb":
            os.environ["VMLX_MINIMAX_MSA"] = "sparse"
            os.environ["VMLX_MINIMAX_MSA_SELECT"] = "reference"
            msa.block_scores = perturbed_scores
        elif m != "auto":
            raise SystemExit(f"unknown mode {m}")

    def lsm(x):
        x = x.astype(mx.float32)
        return x - mx.logsumexp(x, axis=-1, keepdims=True)

    meta = {"default_sort_threshold": default_sort, "sort_threshold": SW.SORT_THRESHOLD, "bundle": a.bundle,
            "vmlx": a.vmlx, "modes": {}}
    for m in a.modes.split(","):
        set_mode(m)
        seq = toks[1:] if m == "shift" else toks
        arrays = {}
        for n in lens:
            picks.clear()
            t0 = time.time()
            cache = model.make_cache()
            lps, ams, tlps = [], [], []
            for s in range(0, n, a.chunk):
                e = min(s + a.chunk, n)
                lp = lsm(model(mx.array(seq[s:e])[None], cache=cache)[0])
                am = mx.argmax(lp, axis=-1).astype(mx.int32)
                tl = mx.take_along_axis(lp, mx.array(seq[s + 1:e + 1])[:, None], axis=-1).reshape(-1)
                if e == n:
                    mx.eval(am, tl, lp)
                    lps.append(lp)
                else:
                    mx.eval(am, tl)
                ams.append(am); tlps.append(tl)
            for t in range(a.decode):
                lp = lsm(model(mx.array([seq[n + t]])[None], cache=cache)[0])
                am = mx.argmax(lp, axis=-1).astype(mx.int32)
                tl = mx.take_along_axis(lp, mx.array([[seq[n + t + 1]]]), axis=-1).reshape(-1)
                mx.eval(am, tl, lp)
                lps.append(lp); ams.append(am); tlps.append(tl)
            trace[f"{m}_{n}"] = list(picks); picks.clear()
            arrays[f"lp_{n}"] = mx.concatenate(lps, axis=0)
            arrays[f"am_{n}"] = mx.concatenate(ams, axis=0)
            arrays[f"tlp_{n}"] = mx.concatenate(tlps, axis=0)
            mx.eval(arrays)
            print(f"{m} n={n}: {time.time() - t0:.1f}s offsets {sorted({c.offset for c in cache})}", flush=True)
            del cache
            mx.clear_cache()
        out = Path(a.out) / f"py_{a.tag}_{m}.safetensors"
        mx.save_safetensors(str(out), arrays)
        meta["modes"][m] = {"file": str(out), "perturbed_calls": perturbed_calls["n"] if m == "perturb" else None}
        print("wrote", out, flush=True)
    (Path(a.out) / f"py_paths_{a.tag}_{'_'.join(a.modes.split(','))}.json").write_text(json.dumps(trace))
    (Path(a.out) / f"py_{a.tag}_meta_{'_'.join(a.modes.split(','))}.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
