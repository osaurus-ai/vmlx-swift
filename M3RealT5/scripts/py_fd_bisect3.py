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
b = mx.array(4, dtype=mx.int32); zero = mx.array(0, dtype=mx.int32)
def composed(q_fn):
    a = x0
    q = q_fn(a)
    rem = mx.remainder(a, b, stream=s1)
    step = mx.logical_and(mx.not_equal(rem, zero, stream=s1),
                          mx.not_equal(mx.less(a, zero, stream=s1), mx.less(b, zero, stream=s1), stream=s1), stream=s1)
    return mx.subtract(q, step.astype(mx.int32, stream=s1), stream=s1)
cases = {
    "composed, q = divmod[0]": lambda: composed(lambda a: mx.divmod(a, b, stream=s1)[0]),
    "composed, q = right_shift": lambda: composed(lambda a: mx.right_shift(a, 2, stream=s1)),
    "floor_divide": lambda: mx.floor_divide(x0, b, stream=s1),
    "floor_divide float32": lambda: mx.floor_divide(x0.astype(mx.float32), 4.0, stream=s1),
    "divide float": lambda: mx.divide(x0, b, stream=s1),
}
for k, f in cases.items():
    print(f"{k:30s} {t(f):.4f} ms/eval")
