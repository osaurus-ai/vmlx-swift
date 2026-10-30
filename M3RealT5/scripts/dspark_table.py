"""Per-prompt table over `M3RealT5 dspark` reports (test harness, 2026-10-29): for each arm (a tag prefix) the plain and
DSpark decode tok/s pooled over its processes (median of every run), DSpark / plain, identity (DSpark == plain and
plain == plain in every run), the greedy token sha per arm, and which arms produced the same greedy text.

  python3 dspark_table.py OUT_DIR LOG_DIR arm1=dspark_q_dspark_default_r arm2=... [--json=out.json]
The greedy sha8 comes from each report's log line (`dspark NAME: ... sha XXXXXXXX`; older builds do not write it to the
JSON); LOG_DIR/<tag>.log for OUT_DIR/dspark_<tag>.json.
"""
import os
import re
import glob
import json
import statistics
import sys

out_dir, log_dir = sys.argv[1], sys.argv[2]
arms, out_json = [], None
for a in sys.argv[3:]:
    if a.startswith("--json="):
        out_json = a.split("=", 1)[1]
    else:
        name, prefix = a.split("=", 1)
        arms.append((name, prefix))
data = {}
for name, prefix in arms:
    files = sorted(glob.glob(f"{out_dir}/{prefix}*.json"))
    if not files:
        raise SystemExit(f"INVALID: no reports for {name} ({prefix})")
    rows = {}
    for f in files:
        tag = os.path.basename(f)[len("dspark_"):-len(".json")]
        shas = dict(re.findall(r"dspark (\S+): .* sha ([0-9a-f]{8})", open(f"{log_dir}/{tag}.log").read()))
        for r in json.load(open(f))["prompts"]:
            d = rows.setdefault(r["name"], {"plain": [], "dspark": [], "ident": True, "sha": set(), "text": r.get("text"),
                                            "accept": [], "tpr": []})
            d["plain"] += r["plain_tok_s"]["runs"]; d["dspark"] += r["dspark_tok_s"]["runs"]
            d["ident"] &= all(x is None for x in r["first_diff_dspark_vs_plain"] + r["first_diff_plain_vs_plain"])
            if r["name"] not in shas:
                raise SystemExit(f"INVALID: no sha for {r['name']} in {tag}.log")
            d["sha"].add(shas[r["name"]])
            for s in r["dspark_stats"]:
                d["accept"].append(s.get("acceptance", 0)); d["tpr"].append(s.get("tokens_per_spec_round", 0))
    data[name] = (rows, len(files))
names = list(dict.fromkeys(n for rows, _ in data.values() for n in rows))
res = {}
hdr = "prompt".ljust(24) + "".join(f"| {a:>8s} plain  dspark   x    acc  ident " for a, _ in arms)
print(hdr)
geo = {a: [] for a, _ in arms}
for n in names:
    line = n[:24].ljust(24)
    res[n] = {}
    for a, _ in arms:
        rows, nf = data[a]
        d = rows.get(n)
        if not d:
            line += "| " + "-".rjust(44)
            continue
        p, s = statistics.median(d["plain"]), statistics.median(d["dspark"])
        x = s / p
        geo[a].append(x)
        res[n][a] = {"plain_median": p, "dspark_median": s, "speedup": x, "runs": len(d["plain"]), "processes": nf,
                     "identity": d["ident"], "sha": sorted(d["sha"]), "acceptance_median": statistics.median(d["accept"]),
                     "tokens_per_round_median": statistics.median(d["tpr"])}
        line += f"| {p:8.2f} {s:7.2f} {x:5.3f} {statistics.median(d['accept']):5.3f} {'ok' if d['ident'] else 'FAIL':>4s}  "
    shas = {a: tuple(res[n][a]["sha"]) for a, _ in arms if a in res[n]}
    groups = {}
    for a, sh in shas.items():
        groups.setdefault(sh, []).append(a)
    line += " same-text: " + " / ".join("+".join(g) for g in groups.values())
    res[n]["same_text_groups"] = list(groups.values())
    print(line)
for a, _ in arms:
    if geo[a]:
        g = statistics.geometric_mean(geo[a])
        print(f"geomean DSpark/plain {a}: {g:.4f} over {len(geo[a])} prompts")
        res.setdefault("_geomean", {})[a] = g
if out_json:
    json.dump(res, open(out_json, "w"), indent=1)
