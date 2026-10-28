import json, sys
s = json.load(open(sys.argv[1])); p = json.load(open(sys.argv[2]))
for size in ("1", "64", "4096", "65536"):
    print("size", size)
    for k, v in s[size].items():
        if isinstance(v, dict):
            q = p[size][k]
            print(f"  {k:28s} swift eval {v['eval_ms_per_call']:.4f} build {v['build_ms_per_call']:.4f} | py eval {q['eval_ms_per_call']:.4f} build {q['build_ms_per_call']:.4f}")
        else:
            print(f"  {k:36s} swift {v:.4f} | py {p[size].get(k, float('nan')):.4f}")
print("cpu swift", s.get("cpu_floorDivide_int32_4096_ms_per_call"))
