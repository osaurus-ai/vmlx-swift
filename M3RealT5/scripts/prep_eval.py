"""Held-out rows for the Swift `evalrows` command (test harness, 2026-10-28): the rows bundle_eval.py scores (set ==
"ref", sorted by id, select_rows(max_tok)), with the source reference's top-128 ids and argmax per position.

  PYTHONPATH=<mm31t parent> python prep_eval.py --corpus C --capture CAP --out OUT_PREFIX [--max-tok N]
Writes OUT_PREFIX.safetensors (ids_i, ti_i, am_i) and OUT_PREFIX.json (row id / domain / effort / n).
"""
import argparse
import json

import numpy as np
from safetensors.numpy import save_file

from mm31t.evalpass import select_rows

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True); ap.add_argument("--capture", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--max-tok", type=int, default=0)
a = ap.parse_args()
ref = np.load(f"{a.capture}/ref_logits.npz")
rows = [json.loads(l) for l in open(a.corpus)]
rows = select_rows(sorted([r for r in rows if r["set"] == "ref"], key=lambda r: r["id"]), a.max_tok)
arrays, manifest = {}, []
for i, r in enumerate(rows):
    ti, am = ref[f"{r['id']}|top_idx"], ref[f"{r['id']}|argmax"]
    assert ti.shape == (len(r["ids"]), 128) and am.shape == (len(r["ids"]),), (r["id"], ti.shape, am.shape)
    arrays[f"ids_{i}"] = np.asarray(r["ids"], dtype=np.int32)
    arrays[f"ti_{i}"] = ti.astype(np.int32)
    arrays[f"am_{i}"] = am.astype(np.int32)
    manifest.append({"id": r["id"], "domain": r["domain"], "effort": r.get("effort"), "n": len(r["ids"])})
save_file(arrays, a.out + ".safetensors")
json.dump({"rows": manifest, "max_tok": a.max_tok, "corpus": a.corpus}, open(a.out + ".json", "w"))
print(len(manifest), "rows", sum(m["n"] for m in manifest), "tokens ->", a.out)
