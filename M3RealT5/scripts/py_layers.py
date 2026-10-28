"""Layer-by-layer dump of the Python vMLX MiniMax-M3.1 forward (test harness): the same decomposition as M3RealT5
`layers`. Keys n{N}_{point} for N in --lens; MoE layers also dump router logits / indices / scores / routed / shared.

  MLX_ENABLE_TF32=0 python py_layers.py --bundle B --vmlx PKG --tokens tokens.safetensors --out F [--lens 100,600,3011]
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True); ap.add_argument("--vmlx", required=True)
    ap.add_argument("--tokens", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--lens", default="100,600,3011"); ap.add_argument("--sort", type=int, default=512)
    ap.add_argument("--uncompiled-router", action="store_true")
    a = ap.parse_args()
    assert os.environ.get("MLX_ENABLE_TF32") == "0"
    sys.path.insert(0, str(Path(a.vmlx).expanduser()))
    import mlx.core as mx
    from vmlx_engine.utils import jang_loader
    import vmlx_engine.jangt.switch as SW
    SW.SORT_THRESHOLD = a.sort
    model, _ = jang_loader.load_jang_model(str(Path(a.bundle).expanduser()))
    inner = model.model
    M = sys.modules[type(inner.layers[0]).__module__]
    ges = M.group_expert_select
    if a.uncompiled_router:
        ges = getattr(ges, "__wrapped__", None) or ges
    toks = mx.load(a.tokens)["tokens"].astype(mx.int32).tolist()
    out = {}
    for n in [int(x) for x in a.lens.split(",")]:
        cache = model.make_cache()
        x = mx.array(toks[:n])[None]
        h = inner.embed_tokens(x)
        out[f"n{n}_embed"] = h
        mask = M.create_attention_mask(h, cache[0])
        for i, (layer, c) in enumerate(zip(inner.layers, cache)):
            xn = layer.input_layernorm(h)
            if n <= 600:
                sa = layer.self_attn
                p0 = f"n{n}_L{i}_"
                q, k, v = sa.project(xn, 0)
                o = mx.fast.scaled_dot_product_attention(q, k, v, scale=sa.scale, mask="causal")
                out.update({p0 + "a_qraw": sa.q_proj(xn), p0 + "a_kraw": sa.k_proj(xn), p0 + "a_vraw": sa.v_proj(xn),
                            p0 + "a_q": q, p0 + "a_k": k, p0 + "a_v": v, p0 + "a_sdpa": o,
                            p0 + "a_out": sa.o_proj(o.transpose(0, 2, 1, 3).reshape(1, n, -1))})
            att = layer.self_attn(xn, mask, c)
            h2 = h + att
            pn = layer.post_attention_layernorm(h2)
            p = f"n{n}_L{i}_"
            if hasattr(layer.mlp, "switch_mlp"):
                moe = layer.mlp
                g = moe.gate(pn)
                inds, scores = ges(g, moe.e_score_correction_bias, moe.top_k, 1, 1, moe.routed_scaling_factor,
                                   moe.norm_topk_prob)
                routed = moe.routed(pn, inds, scores)
                shared = moe.shared_experts(pn)
                m = (routed + shared).astype(pn.dtype)
                out.update({p + "router_logits": g, p + "inds": inds.astype(mx.int32), p + "scores": scores,
                            p + "routed": routed, p + "shared": shared})
                full = layer.mlp(pn)
                out[p + "mlp_module"] = full
            else:
                m = layer.mlp(pn)
            hn = h2 + m
            out.update({p + "in_norm": xn, p + "attn": att, p + "h_mid": h2, p + "post_norm": pn, p + "mlp": m,
                        p + "h_out": hn})
            mx.eval(out)
            h = hn
        fn = inner.norm(h)
        out[f"n{n}_final_norm"] = fn
        out[f"n{n}_logits"] = model.lm_head(fn)
        mx.eval(out)
        print("n", n, "done", flush=True)
    mx.save_safetensors(a.out, out)
    print("wrote", a.out, len(out))


if __name__ == "__main__":
    main()
