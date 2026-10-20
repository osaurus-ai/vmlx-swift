#!/usr/bin/env python3
"""Build production metadata, geometry, CPU pixels and XCTest contracts without MLX.

Uses the unmodified Foundation-only Kit files and Flux3 sources in a temporary
SwiftPM package. Only FluxAdapter's value declaration is extracted: its runtime
implementation imports MLX. No substitute validator, routing or math is used.
Registry tests use the production entry factory with a loader that is never invoked.
This does NOT validate the native tensor model or the complete Swift package.
"""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

repo = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix="flux3-contracts-") as temp:
    root = Path(temp)
    kit = root / "Sources/vMLXFluxKit"
    models = root / "Sources/vMLXFluxModels"
    tests = root / "Tests/Flux3ContractsTests"
    for path in [kit, models, tests]:
        path.mkdir(parents=True)
    for name in ["Requests", "EffectiveImageParameters", "FluxModel", "FluxInterrupt", "ImageBatchPlan", "FluxModelLimits", "LocalModelStore", "ModelRegistry", "SeedVR2BundleMetadata"]:
        shutil.copy(repo / f"Libraries/vMLXFluxKit/{name}.swift", kit)
    adapter = (repo / "Libraries/vMLXFluxKit/FluxAdapter.swift").read_text()
    value = adapter[adapter.index("public struct FluxAdapter:"):adapter.index("/// Installed only")]
    (kit / "FluxAdapter.swift").write_text("import Foundation\n" + value)
    for name in ["Flux3Bundle", "Flux3Schedule", "Flux3ReferencePlan", "Flux3Registration", "Flux3Math", "Flux3Pixels", "Flux3ImageReader", "Flux3WeightManifest"]:
        shutil.copy(repo / f"Libraries/vMLXFluxModels/Flux3Klein/{name}.swift", models)
    shutil.copy(repo / "Tests/vMLXFluxTests/Flux3KleinTests.swift", tests)
    shutil.copytree(repo / "Tests/vMLXFluxTests/Resources", tests / "Resources")
    (root / "Package.swift").write_text('''// swift-tools-version: 6.1
import PackageDescription
let package = Package(name: "Flux3Contracts", platforms: [.macOS(.v14)], targets: [
    .target(name: "vMLXFluxKit"),
    .target(name: "vMLXFluxModels", dependencies: ["vMLXFluxKit"]),
    .testTarget(name: "Flux3ContractsTests", dependencies: ["vMLXFluxKit", "vMLXFluxModels"],
                resources: [.copy("Resources")])
])
''')
    env = os.environ.copy()
    env.setdefault("DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer")
    env["CLANG_MODULE_CACHE_PATH"] = str(root / "clang-cache")
    env["SWIFTPM_MODULECACHE_OVERRIDE"] = str(root / "swift-cache")
    subprocess.run(["swift", "test", "--disable-sandbox", "--package-path", str(root), "--filter", "Flux3KleinTests"], env=env, check=True)
