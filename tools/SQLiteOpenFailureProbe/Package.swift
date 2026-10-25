// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "SQLiteOpenFailureProbe",
    platforms: [.macOS(.v14)],
    dependencies: [.package(name: "RuntimeUnderTest", path: "../..")],
    targets: [
        .target(name: "SQLiteProbeSupport", linkerSettings: [.linkedLibrary("sqlite3")]),
        .executableTarget(
            name: "SQLiteOpenFailureProbe",
            dependencies: [
                .product(name: "MLXLMCommon", package: "RuntimeUnderTest"), "SQLiteProbeSupport",
            ]),
    ])
