"""First-divergence report between two layer dumps (py_layers.py / M3RealT5 layers)."""
import re, sys
import numpy as np
from safetensors import safe_open

def load(p):
    out = {}
    with safe_open(p, framework="numpy") as f:
        for k in f.keys():
            try:
                out[k] = f.get_tensor(k)
            except Exception:
                pass
    return out

def load_mx(p):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    d = mx.load(p)
    return {k: (np.array(v.view(mx.uint16)) if v.dtype == mx.bfloat16 else np.array(v), str(v.dtype)) for k, v in d.items()}

A, B = load_mx(sys.argv[1]), load_mx(sys.argv[2])
order = ["embed", "in_norm", "a_qraw", "a_kraw", "a_vraw", "a_q", "a_k", "a_v", "a_sdpa", "a_out", "attn", "h_mid", "post_norm", "router_logits", "inds", "scores", "routed", "shared", "mlp", "mlp_module", "h_out", "final_norm", "logits"]
def key_order(k):
    m = re.match(r"n(\d+)_(?:L(\d+)_)?(.*)", k)
    n, L, pt = int(m.group(1)), int(m.group(2)) if m.group(2) else (-1 if m.group(3) == "embed" else 99), m.group(3)
    return (n, L, order.index(pt) if pt in order else 50)
for k in sorted(set(A) | set(B), key=key_order):
    if k not in A or k not in B:
        print(f"{k:28s} missing in {'A' if k not in A else 'B'}"); continue
    (a, da), (b, db) = A[k], B[k]
    if a.shape != b.shape or da != db:
        print(f"{k:28s} shape/dtype {a.shape} {da} vs {b.shape} {db}"); continue
    same = np.array_equal(a, b)
    if da == "mlx.core.bfloat16":
        fa = (a.astype(np.uint32) << 16).view(np.float32); fb = (b.astype(np.uint32) << 16).view(np.float32)
    else:
        fa, fb = a.astype(np.float64), b.astype(np.float64)
    diff = np.abs(fa.astype(np.float64) - fb.astype(np.float64))
    nd = int((a != b).sum())
    rel = diff.max() / (np.abs(fa).max() + 1e-30)
    print(f"{k:28s} {da.split('.')[-1]:9s} {'IDENTICAL' if same else 'differ'} n_diff={nd}/{a.size} max_abs={diff.max():.3g} rel_to_max={rel:.3g}")
