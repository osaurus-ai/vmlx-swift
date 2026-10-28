"""Bisect which piece of integer floor_divide's composition costs extra on a non-default stream."""
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
four, zero = mx.array(4, dtype=mx.int32), mx.array(0, dtype=mx.int32)
a = lambda: mx.add(x0, four, stream=s1)
cases = {
    "baseline a": lambda: a(),
    "a + less(scalar,scalar)": lambda: mx.add(a(), mx.less(four, zero, stream=s1).astype(mx.int32, stream=s1), stream=s1),
    "a + less(vec,0)": lambda: mx.add(a(), mx.less(a(), zero, stream=s1).astype(mx.int32, stream=s1), stream=s1),
    "divmod quotient": lambda: mx.divmod(a(), four, stream=s1)[0],
    "a // 4 (floor_divide)": lambda: mx.floor_divide(a(), four, stream=s1),
    "remainder": lambda: mx.remainder(a(), four, stream=s1),
    "not_equal(less, less_scalar)": lambda: mx.not_equal(mx.less(a(), zero, stream=s1), mx.less(four, zero, stream=s1), stream=s1),
}
for k, f in cases.items():
    print(f"{k:30s} {t(f):.4f} ms/eval")
