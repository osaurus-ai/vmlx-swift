"""Does integer floor_divide on a non-default GPU stream cost extra in Python too? Per-eval ms, median of 5 x 300."""
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
four, three, two = mx.array(4, dtype=mx.int32), mx.array(3, dtype=mx.int32), mx.array(2, dtype=mx.int32)
cases = {
    "fd default stream": lambda: mx.floor_divide(x0 * four + three, four),
    "fd new stream s1": lambda: mx.floor_divide(mx.add(mx.multiply(x0, four, stream=s1), three, stream=s1), four, stream=s1),
    "rshift new stream s1": lambda: mx.right_shift(mx.add(mx.multiply(x0, four, stream=s1), three, stream=s1), two, stream=s1),
    "remainder s1": lambda: mx.remainder(x0, four, stream=s1),
    "less s1": lambda: mx.less(x0, mx.array(0, dtype=mx.int32), stream=s1),
}
for k, f in cases.items():
    print(f"{k:24s} {t(f):.4f} ms/eval")
with mx.stream(s1):
    print(f"{'fd s1 as default':24s} {t(lambda: mx.floor_divide(x0 * four + three, four)):.4f} ms/eval")
