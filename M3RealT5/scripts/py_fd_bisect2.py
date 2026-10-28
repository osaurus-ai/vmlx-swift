import statistics, time
import mlx.core as mx
x0 = mx.arange(4096, dtype=mx.int32); mx.eval(x0)
s1 = mx.new_stream(mx.gpu)
def t(fn, n=300, probes=5):
    r = []
    for p in range(probes + 1):
        t0 = time.perf_counter()
        for _ in range(n): mx.eval(fn())
        if p: r.append((time.perf_counter() - t0) / n * 1e3)
    return statistics.median(r)
four = mx.array(4, dtype=mx.int32); fourv = mx.full((4096,), 4, dtype=mx.int32); mx.eval(four, fourv)
four_s1 = mx.multiply(four, 1, stream=s1); mx.eval(four_s1)
cases = {
    "fd(vec x0, scalar 4) s1": lambda: mx.floor_divide(x0, four, stream=s1),
    "fd(vec x0, vec 4) s1": lambda: mx.floor_divide(x0, fourv, stream=s1),
    "fd(vec x0, scalar 4) default": lambda: mx.floor_divide(x0, four),
    "fd(vec x0, vec 4) default": lambda: mx.floor_divide(x0, fourv),
    "fd(x0, four made on s1) s1": lambda: mx.floor_divide(x0, four_s1, stream=s1),
}
for k, f in cases.items():
    print(f"{k:34s} {t(f):.4f} ms/eval")
