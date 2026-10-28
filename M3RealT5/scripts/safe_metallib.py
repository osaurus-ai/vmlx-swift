"""Rebuild the SwiftPM build's precompiled MLX metallib with MLX's own Metal flags (-fno-fast-math) and assemble a
runnable copy of M3RealT5 that loads it (test harness, 2026-10-28).

SwiftPM compiles Source/Cmlx/mlx-generated/metal/*.metal (rope, rms_norm, sdpa, layer_norm, ...) with Xcode's default
MTL_FAST_MATH (-fmetal-math-mode=fast -fmetal-math-fp32-functions=fast); MLX's CMake (the Python wheel) uses
-fno-fast-math. The exe loads mlx-swift_Cmlx.bundle from its own directory first, so a copied exe + bundles with the
rebuilt metallib runs the same code with MLX's arithmetic.

  python3 safe_metallib.py <worktree> <out_dir>
"""
import glob, json, os, shlex, shutil, subprocess, sys

tree, out = sys.argv[1], os.path.abspath(sys.argv[2])
rel = os.path.join(tree, ".build/out/Products/Release")
air = os.path.join(out, "air"); os.makedirs(air, exist_ok=True)
manifest = max(glob.glob(os.path.join(tree, ".build/out/Intermediates.noindex/XCBuildData/*.xcbuilddata/manifest.json")),
               key=os.path.getmtime)
cmds = json.load(open(manifest))["commands"]
n = 0
for v in cmds.values():
    a = v.get("args")
    if not a or not str(a[0]).endswith("/metal") or "-c" not in a:
        continue
    b = [x for x in a if x not in ("-fmetal-math-mode=fast", "-fmetal-math-fp32-functions=fast")]
    i = b.index("-serialize-diagnostics"); del b[i:i + 2]
    i = b.index("-MMD"); del b[i:i + 5]
    o = b.index("-o"); b[o + 1] = os.path.join(air, os.path.basename(b[-1]).replace(".metal", ".air"))
    b.insert(1, "-fno-fast-math")
    subprocess.run(b, check=True, stderr=subprocess.DEVNULL); n += 1
lib = os.path.join(out, "default.metallib")
subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-target", "air64-apple-macos14.0", "-o", lib]
               + sorted(glob.glob(os.path.join(air, "*.air"))), check=True)
bin_dir = os.path.join(out, "bin"); shutil.rmtree(bin_dir, ignore_errors=True); os.makedirs(bin_dir)
shutil.copy2(os.path.join(rel, "M3RealT5"), bin_dir)
for bundle in glob.glob(os.path.join(rel, "*.bundle")):
    shutil.copytree(bundle, os.path.join(bin_dir, os.path.basename(bundle)))
shutil.copy2(lib, os.path.join(bin_dir, "mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"))
print(f"{n} metal files rebuilt with -fno-fast-math -> {bin_dir}/M3RealT5")
