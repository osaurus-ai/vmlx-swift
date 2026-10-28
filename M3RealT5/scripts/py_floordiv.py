"""Python side of M3RealT5 floordiv: per-call cost of integer floor_divide vs a shift and the composed op's primitives
(dependent chains of --iters calls, eval-timed; build = graph construction only), median over probes after 1 discarded."""
import argparse, json, statistics, time
import mlx.core as mx

ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=50); ap.add_argument("--probes", type=int, default=7)
ap.add_argument("--out", required=True); a = ap.parse_args()
four, three, two, zero, one = (mx.array(v, dtype=mx.int32) for v in (4, 3, 2, 0, 1))
ops = [
    ("floorDivide_int32", lambda x: mx.floor_divide(x * four + three, four)),
    ("rightShift_int32", lambda x: mx.right_shift(x * four + three, two)),
    ("method_floorDivide_int32", lambda x: (x * four + three) // 4),
    ("remainder_int32", lambda x: x + mx.remainder(x, four) * zero),
    ("less_int32", lambda x: x + (x < zero).astype(mx.int32)),
    ("notEqual_int32", lambda x: x + (x != x).astype(mx.int32)),
    ("logicalAnd", lambda x: x + mx.logical_and(x < zero, x < zero).astype(mx.int32)),
    ("subtract_int32", lambda x: x - zero),
    ("floorDivide_float32", lambda x: mx.floor_divide((x * four + three).astype(mx.float32), mx.array(4.0)).astype(mx.int32)),
    ("baseline_mul_add", lambda x: x * one + zero),
]
now = lambda: time.perf_counter() * 1e3
rep = {"iters": a.iters, "probes": a.probes}
for size in (1, 64, 4096, 65536):
    x0 = mx.arange(size, dtype=mx.int32); mx.eval(x0)
    ev, bd = {}, {}
    for p in range(a.probes + 1):
        for name, op in (ops if p % 2 == 0 else ops[::-1]):
            mx.synchronize()
            t0 = now(); x = x0
            for _ in range(a.iters): x = op(x)
            t1 = now(); mx.eval(x); t2 = now()
            if p:
                bd.setdefault(name, []).append((t1 - t0) / a.iters); ev.setdefault(name, []).append((t2 - t1) / a.iters)
            assert mx.array_equal(x, x0).item(), name
    row = {n: {"eval_ms_per_call": statistics.median(ev[n]), "build_ms_per_call": statistics.median(bd[n])} for n, _ in ops}
    single = {}
    for p in range(a.probes + 1):
        for name, op in ops:
            if name not in ("floorDivide_int32", "rightShift_int32", "baseline_mul_add"): continue
            t = []
            for _ in range(20):
                t0 = now(); y = op(x0); mx.eval(y); t.append(now() - t0)
            if p: single.setdefault(name, []).append(statistics.median(t))
    for n, v in single.items(): row[f"single_eval_ms_{n}"] = statistics.median(v)
    rep[str(size)] = row
    print("size", size, json.dumps({k: (round(v["eval_ms_per_call"], 4) if isinstance(v, dict) else round(v, 4)) for k, v in row.items()}), flush=True)
json.dump(rep, open(a.out, "w"), indent=1)
