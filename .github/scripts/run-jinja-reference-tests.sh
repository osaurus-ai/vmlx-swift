#!/usr/bin/env bash
set -euo pipefail

# Compile the real vendored interpreter and reference tests without an unrelated
# MLX/Metal build. Symlinks keep the source under test identical to the package.
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
test_root=$(mktemp -d "${TMPDIR:-/tmp}/vmlx-jinja-tests.XXXXXX")
trap 'rm -rf "$test_root"' EXIT
mkdir -p "$test_root/Sources/MLXLMCommon" "$test_root/Tests"
ln -s "$repo_root/Libraries/MLXLMCommon/Tool/ToolSchemaForChatTemplate.swift" "$test_root/Sources/MLXLMCommon/ToolSchemaForChatTemplate.swift"
ln -s "$repo_root/Tests/ToolSchemaReferenceTests" "$test_root/Tests/ToolSchemaReferenceTests"
ln -s "$repo_root/Vendors/Jinja/Sources/Jinja" "$test_root/Sources/VMLXJinja"
ln -s "$repo_root/Tests/JinjaReferenceTests" "$test_root/Tests/JinjaReferenceTests"
cat > "$test_root/Package.swift" <<'SWIFT'
// swift-tools-version: 6.1
import PackageDescription
let package = Package(
    name: "JinjaReferenceControls",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.5.0")
    ],
    targets: [
        .target(name: "MLXLMCommon"),
        .testTarget(name: "ToolSchemaReferenceTests", dependencies: ["MLXLMCommon", "VMLXJinja"]),
        .target(name: "VMLXJinja", dependencies: [
            .product(name: "OrderedCollections", package: "swift-collections")
        ]),
        .testTarget(name: "JinjaReferenceTests", dependencies: ["VMLXJinja"],
            exclude: ["README.md"], resources: [.copy("Fixtures")])
    ])
SWIFT
swift test --package-path "$test_root" --jobs 4
