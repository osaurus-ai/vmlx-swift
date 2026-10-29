"""Paired comparison of two Swift held-out dumps (`evalrows` / `evalturns`) against the source reference (test harness,
2026-10-29). Same KL as score_eval.py (reference top-128 + tail bucket), top-1 vs the reference argmax. Per arm the
assistant-position means; paired B - A differences with a 95 % row-bootstrap interval (rows resampled, 2,000 draws);
identity: positions / rows whose log-probs are equal; the same split by domain and by rows whose prefill had a
non-final piece of 18-127 rows (`--info` = the evalturns JSON of either arm, its `row_info`).

  PYTHONPATH=<mm31t parent> python pair_eval.py --in IN_PREFIX --capture CAP --a 'OUT/evalturns_X_*.safetensors' \
      --b 'OUT/evalturns_Y_*.safetensors' [--info OUT/evalturns_X.json] --out pair.json
"""
import argparse
import glob
import json

import numpy as np
from safetensors.numpy import load_file

from mm31t.evalpass import assistant_targets

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--capture", required=True)
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--info")
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
man = json.load(open(args.inp + ".json"))["rows"]
inp = load_file(args.inp + ".safetensors")


def load(pattern):
    d = {}
    for f in sorted(glob.glob(pattern)):
        d.update(load_file(f))
    if not d:
        raise SystemExit(f"INVALID: no dumps match {pattern}")
    return d


A, B = load(args.a), load(args.b)
rows = [i for i in range(len(man)) if f"q_{i}" in A and f"q_{i}" in B]
if not rows:
    raise SystemExit("INVALID: no common rows")
affected = None
if args.info:
    info = json.load(open(args.info))["row_info"]
    affected = {r["i"]: sum(s["affected"] for s in r["segments"]) > 0 for r in info}
ref = np.load(f"{args.capture}/ref_logits.npz")


def kl_rows(tlp, q):
    p_top = np.exp(tlp)
    tail_p = np.maximum(1.0 - p_top.sum(-1), 1e-12); tail_q = np.maximum(1.0 - np.exp(q).sum(-1), 1e-12)
    return (p_top * (tlp - q)).sum(-1) + tail_p * (np.log(tail_p) - np.log(tail_q))


per = []  # per row: domain, affected, asst count, sums
for i in rows:
    m = man[i]
    ids = inp[f"ids_{i}"].tolist()
    tlp = ref[f"{m['id']}|top_lp"].astype(np.float64)
    qa, qb = A[f"q_{i}"].astype(np.float64), B[f"q_{i}"].astype(np.float64)
    assert qa.shape == tlp.shape == qb.shape, m["id"]
    if not (np.isfinite(qa).all() and np.isfinite(qb).all()):
        raise SystemExit(f"INVALID non-finite log-probs in {m['id']}")
    msk = assistant_targets(ids)
    ka, kb = kl_rows(tlp, qa)[msk], kl_rows(tlp, qb)[msk]
    ta, tb = (A[f"rank_{i}"] == 0)[msk], (B[f"rank_{i}"] == 0)[msk]
    same_pos = (qa == qb).all(-1)
    per.append({"i": i, "domain": m["domain"], "affected": None if affected is None else affected.get(i),
                "n": int(msk.sum()), "kl_a": float(ka.sum()), "kl_b": float(kb.sum()),
                "t1_a": int(ta.sum()), "t1_b": int(tb.sum()), "same_rows": bool(same_pos.all()),
                "same_pos": int(same_pos.sum()), "pos": len(ids), "flips": int((ta != tb).sum())})

rng = np.random.default_rng(args.seed)


def summary(sel):
    if not sel:
        return {"rows": 0}
    n = np.array([r["n"] for r in sel], float)
    ka, kb = np.array([r["kl_a"] for r in sel]), np.array([r["kl_b"] for r in sel])
    ta, tb = np.array([r["t1_a"] for r in sel], float), np.array([r["t1_b"] for r in sel], float)
    tot = n.sum()
    out = {"rows": len(sel), "asst_positions": int(tot), "kl_a": ka.sum() / tot, "kl_b": kb.sum() / tot,
           "top1_a": ta.sum() / tot, "top1_b": tb.sum() / tot, "rows_identical": sum(r["same_rows"] for r in sel),
           "positions_identical_frac": sum(r["same_pos"] for r in sel) / sum(r["pos"] for r in sel),
           "top1_flips": sum(r["flips"] for r in sel)}
    out["d_kl"] = out["kl_b"] - out["kl_a"]; out["d_top1"] = out["top1_b"] - out["top1_a"]
    idx = rng.integers(0, len(sel), size=(2000, len(sel)))
    nn = n[idx].sum(1)
    dk = (kb[idx].sum(1) - ka[idx].sum(1)) / nn
    dt = (tb[idx].sum(1) - ta[idx].sum(1)) / nn
    out["d_kl_ci95"] = [float(np.quantile(dk, 0.025)), float(np.quantile(dk, 0.975))]
    out["d_top1_ci95"] = [float(np.quantile(dt, 0.025)), float(np.quantile(dt, 0.975))]
    return out


res = {"a": args.a, "b": args.b, "all": summary(per), "domains": {}}
for d in sorted({r["domain"] for r in per}):
    res["domains"][d] = summary([r for r in per if r["domain"] == d])
if affected is not None:
    res["affected_rows"] = summary([r for r in per if r["affected"]])
    res["unaffected_rows"] = summary([r for r in per if r["affected"] is False])
json.dump(res, open(args.out, "w"), indent=1)


def line(name, s):
    if not s.get("rows"):
        return
    print(f"{name:16s} rows {s['rows']:4d} asst {s['asst_positions']:7d} KL {s['kl_a']:.4f} -> {s['kl_b']:.4f} "
          f"(d {s['d_kl']:+.5f}, CI {s['d_kl_ci95'][0]:+.5f}..{s['d_kl_ci95'][1]:+.5f}) top1 {100 * s['top1_a']:.2f} -> "
          f"{100 * s['top1_b']:.2f} % (d {100 * s['d_top1']:+.3f} pp, CI {100 * s['d_top1_ci95'][0]:+.3f}.."
          f"{100 * s['d_top1_ci95'][1]:+.3f}) identical rows {s['rows_identical']} pos {100 * s['positions_identical_frac']:.1f} %"
          f" flips {s['top1_flips']}")


line("ALL", res["all"])
for d, s in res["domains"].items():
    line(d, s)
if affected is not None:
    line("affected", res["affected_rows"]); line("unaffected", res["unaffected_rows"])
