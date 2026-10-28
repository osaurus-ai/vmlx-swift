import statistics, time
import mlx.core as mx
x0 = mx.arange(4096, dtype=mx.int32); mx.eval(x0)
xf = x0.astype(mx.float32); mx.eval(xf)
s1 = mx.new_stream(mx.gpu)
def t(fn, n=300, probes=5):
    r = []
    for p in range(probes + 1):
        t0 = time.perf_counter()
        for _ in range(n): mx.eval(fn())
        if p: r.append((time.perf_counter() - t0) / n * 1e3)
    return statistics.median(r)
b = mx.array(4, dtype=mx.int32); bf = mx.array(4.0)
cases = {
    "floor(divide) float s1": lambda: mx.floor(mx.divide(xf, bf, stream=s1), stream=s1),
    "floor_divide float s1": lambda: mx.floor_divide(xf, bf, stream=s1),
    "floor_divide float s1 (op only, x on s1)": lambda: mx.floor_divide(mx.add(xf, 0.0, stream=s1), bf, stream=s1),
    "floor_divide int s1": lambda: mx.floor_divide(x0, b, stream=s1),
    "floor_divide int default": lambda: mx.floor_divide(x0, b),
    "floor_divide int, stream=mx.gpu": lambda: mx.floor_divide(x0, b, stream=mx.gpu),
}
for k, f in cases.items():
    print(f"{k:42s} {t(f):.4f} ms/eval")
