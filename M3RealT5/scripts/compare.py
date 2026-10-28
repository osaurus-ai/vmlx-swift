"""Compare two parity dumps (Swift / Python M3RealT5 format): per length, over the last chunk + decode rows (lp_N) and
over every position (am_N argmax, tlp_N next-token log-prob). P = first file (reference), Q = second.

  python compare.py A.safetensors B.safetensors [--decode 16] [--json out.json] [--name label]
Fails closed (INVALID) on missing keys, shape mismatch, empty or non-finite reference.
"""
import argparse
import json
import sys

import numpy as np
from safetensors.numpy import load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--json"); ap.add_argument("--name", default="")
    a = ap.parse_args()
    A, B = load_file(a.a), load_file(a.b)
    lens = sorted({int(k.split("_")[1]) for k in A if k.startswith("lp_")})
    res = {"a": a.a, "b": a.b, "name": a.name, "lens": {}}
    for n in lens:
        r = {}
        for k in ("lp", "am", "tlp"):
            key = f"{k}_{n}"
            if key not in A or key not in B or A[key].shape != B[key].shape or A[key].size == 0:
                print("INVALID", key, A.get(key, np.zeros(0)).shape, B.get(key, np.zeros(0)).shape)
                sys.exit(2)
        P, Q = A[f"lp_{n}"].astype(np.float64), B[f"lp_{n}"].astype(np.float64)
        if not (np.isfinite(P).all() and np.isfinite(Q).all()):
            print("INVALID non-finite"); sys.exit(2)
        d = np.abs(P - Q)
        pp = np.exp(P)
        kl = (pp * (P - Q)).sum(-1)
        top1 = (P.argmax(-1) == Q.argmax(-1))
        nd = a.decode
        mass = pp > 1e-4
        r["rows"] = int(P.shape[0])
        r["bit_identical"] = bool(np.array_equal(A[f"lp_{n}"], B[f"lp_{n}"]))
        r["max_abs_dlogprob"] = float(d.max())
        r["max_abs_dlogprob_p_gt_1e-4"] = float(d[mass].max()) if mass.any() else 0.0
        r["kl_mean"] = float(kl.mean()); r["kl_max"] = float(kl.max())
        r["top1"] = float(top1.mean())
        r["chunk_rows"] = int(P.shape[0] - nd)
        r["chunk_kl_mean"] = float(kl[:-nd].mean()); r["decode_kl_mean"] = float(kl[-nd:].mean())
        r["chunk_top1"] = float(top1[:-nd].mean()); r["decode_top1"] = float(top1[-nd:].mean())
        r["decode_max_abs_dlogprob"] = float(d[-nd:].max())
        am_a, am_b = A[f"am_{n}"], B[f"am_{n}"]
        ta, tb = A[f"tlp_{n}"].astype(np.float64), B[f"tlp_{n}"].astype(np.float64)
        r["all_positions"] = int(am_a.size)
        r["all_top1"] = float((am_a == am_b).mean())
        r["all_top1_mismatches"] = int((am_a != am_b).sum())
        r["all_max_abs_dtarget_logprob"] = float(np.abs(ta - tb).max())
        r["all_mean_abs_dtarget_logprob"] = float(np.abs(ta - tb).mean())
        r["all_target_bit_identical"] = bool(np.array_equal(A[f"tlp_{n}"], B[f"tlp_{n}"]))
        r["nll_a"] = float(-ta.mean()); r["nll_b"] = float(-tb.mean())
        res["lens"][str(n)] = r
        print(a.name, n, json.dumps(r))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
