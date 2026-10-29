"""Score the Swift `evalrows` dump exactly as bundle_eval.py scores vMLX Python (test harness, 2026-10-28): KL over the
reference's top-128 support plus one tail bucket, top-1 / top-5 / top-10 vs the reference argmax, NLL vs the reference,
per domain over all / assistant positions, the adaptive think / skip gate at assistant turn starts. Optionally prints
the same numbers from a Python bundle_eval JSON (--python) side by side.

  PYTHONPATH=<mm31t parent> python score_eval.py --in IN_PREFIX --swift 'OUT/evalrows_TAG_*.safetensors' \
      --capture CAP --out score.json [--python final_eval_r8.json]
"""
import argparse
import glob
import json

import numpy as np
from safetensors.numpy import load_file

from mm31t.capture import SKIP, THINK, decisions
from mm31t.evalpass import assistant_targets

BINS = (0.0, 0.3, 0.5, 0.7, 0.9, 0.99, 1.01)


def stats(kl, t1, rank, tail):
    if len(kl) == 0:
        return {"n": 0}
    q = np.quantile(kl, [0.5, 0.9, 0.99, 0.999])
    return {"n": int(len(kl)), "kl_mean": float(kl.mean()), "kl_median": float(q[0]), "kl_p90": float(q[1]),
            "kl_p99": float(q[2]), "kl_p999": float(q[3]), "kl_max": float(kl.max()), "top1": float(t1.mean()),
            "top5": float((rank < 5).mean()), "top10": float((rank < 10).mean()), "tail_mass": float(tail.mean())}


ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--swift", required=True)
ap.add_argument("--capture", required=True); ap.add_argument("--out", required=True); ap.add_argument("--python")
a = ap.parse_args()
man = json.load(open(a.inp + ".json"))["rows"]
inp = load_file(a.inp + ".safetensors")
sw = {}
for f in sorted(glob.glob(a.swift)):
    sw.update(load_file(f))
missing = [i for i in range(len(man)) if f"q_{i}" not in sw]
if missing:
    raise SystemExit(f"INVALID: {len(missing)} rows have no Swift output (first {missing[:3]})")
ref = np.load(f"{a.capture}/ref_logits.npz")
ref_nll = json.load(open(f"{a.capture}/nll.json"))
ref_dec = {(d["id"], d["pos"]): d for d in map(json.loads, open(f"{a.capture}/decisions.jsonl")) if d["set"] == "ref"}
acc, dec, nlls = {}, [], []
for i, m in enumerate(man):
    ids = inp[f"ids_{i}"].tolist()
    n = len(ids)
    tlp = ref[f"{m['id']}|top_lp"].astype(np.float64)
    q = sw[f"q_{i}"].astype(np.float64)
    assert q.shape == tlp.shape and sw[f"rank_{i}"].shape == (n,) and sw[f"tl_{i}"].shape == (n - 1,), m["id"]
    if not np.isfinite(q).all():
        raise SystemExit(f"INVALID non-finite Swift log-probs in {m['id']}")
    p_top = np.exp(tlp)
    tail_p = np.maximum(1.0 - p_top.sum(-1), 1e-12); tail_q = np.maximum(1.0 - np.exp(q).sum(-1), 1e-12)
    kl = (p_top * (tlp - q)).sum(-1) + tail_p * (np.log(tail_p) - np.log(tail_q))
    rank = sw[f"rank_{i}"]
    top_p = np.exp(tlp.max(axis=1))
    acc.setdefault(m["domain"], []).append((kl, rank == 0, rank, tail_p, assistant_targets(ids), top_p))
    nlls.append({"id": m["id"], "domain": m["domain"], "nll": float(-sw[f"tl_{i}"].astype(np.float64).mean()),
                 "ref_nll": ref_nll.get(m["id"])})
    th, sk = sw[f"th_{i}"], sw[f"sk_{i}"]
    for p in decisions(ids):
        if (m["id"], p) in ref_dec:
            d = ref_dec[(m["id"], p)]
            dec.append({"domain": m["domain"], "effort": m["effort"], "ref": [d["lp_think"], d["lp_skip"]],
                        "q": [float(th[p - 1]), float(sk[p - 1])]})
res = {"rows": len(man), "domains": {}, "by_ref_top1": {}}
allp = []
for d, parts in sorted(acc.items()):
    cat = [np.concatenate([p[k] for p in parts]) for k in range(6)]
    msk = cat[4]
    res["domains"][d] = {"all": stats(cat[0], cat[1], cat[2], cat[3]),
                         "asst": stats(cat[0][msk], cat[1][msk], cat[2][msk], cat[3][msk])}
    allp.append(cat)
cat = [np.concatenate([p[k] for p in allp]) for k in range(6)]
res["all"] = stats(cat[0], cat[1], cat[2], cat[3])
res["asst"] = stats(cat[0][cat[4]], cat[1][cat[4]], cat[2][cat[4]], cat[3][cat[4]])
for lo, hi in zip(BINS[:-1], BINS[1:]):
    msk = (cat[5] >= lo) & (cat[5] < hi)
    res["by_ref_top1"][f"{lo:.2f}-{min(hi, 1.0):.2f}"] = stats(cat[0][msk], cat[1][msk], cat[2][msk], cat[3][msk])
dn = [x for x in nlls if x["ref_nll"] is not None]
res["nll"] = {"bundle": float(np.mean([x["nll"] for x in dn])), "ref": float(np.mean([x["ref_nll"] for x in dn]))}


def gate(ds):
    rt = np.array([x["ref"][0] > x["ref"][1] for x in ds]); qt = np.array([x["q"][0] > x["q"][1] for x in ds])
    pr, pq = np.exp([x["ref"][0] for x in ds]), np.exp([x["q"][0] for x in ds])
    return {"n": len(ds), "agree": float((rt == qt).mean()), "ref_think": float(rt.mean()), "q_think": float(qt.mean()),
            "mean_abs_dp_think": float(np.abs(pr - pq).mean())}


if dec:
    res["decision"] = {"all": gate(dec), "by_domain": {d: gate([x for x in dec if x["domain"] == d])
                                                       for d in sorted({x["domain"] for x in dec})}}
json.dump(res, open(a.out, "w"), indent=1)
py = json.load(open(a.python)) if a.python else None
print(f"{'domain':14s} {'asst n':>8s} {'KL swift':>9s} {'KL py':>8s} {'top1 sw':>8s} {'top1 py':>8s} {'top5 sw':>8s} {'top5 py':>8s}")
for d in sorted(res["domains"]) + ["ALL"]:
    s = res["asst"] if d == "ALL" else res["domains"][d]["asst"]
    p = (py["asst"] if d == "ALL" else py["domains"].get(d, {}).get("asst", {})) if py else {}
    if s.get("n", 0) == 0:
        continue
    print(f"{d:14s} {s['n']:8d} {s['kl_mean']:9.4f} {p.get('kl_mean', float('nan')):8.4f} {100 * s['top1']:8.2f} "
          f"{100 * p.get('top1', float('nan')):8.2f} {100 * s['top5']:8.2f} {100 * p.get('top5', float('nan')):8.2f}")
print("all positions KL", res["all"]["kl_mean"], "top1", res["all"]["top1"], "| python",
      py["all"]["kl_mean"] if py else None, py["all"]["top1"] if py else None)
print("NLL swift", res["nll"], "| python", py["nll"]["bundle"] if py else None)
print("decision", res.get("decision", {}).get("all"), "| python", (py or {}).get("decision", {}).get("all"))
