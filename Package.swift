// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.
// Copyright © 2024 Apple Inc.

import CompilerPluginSupport
import PackageDescription

#if os(Linux)
    let platformExcludes: [String] = [
        // Linux specific excludes
        "framework",
        "include-framework",
        "metal-cpp",
        // Exclude Metal backend files on Linux, but keep no_metal.cpp for stubs
        "mlx/mlx/backend/metal/allocator.cpp",
        "mlx/mlx/backend/metal/binary.cpp",
        "mlx/mlx/backend/metal/compiled.cpp",
        "mlx/mlx/backend/metal/conv.cpp",
        "mlx/mlx/backend/metal/copy.cpp",
        "mlx/mlx/backend/metal/custom_kernel.cpp",
        "mlx/mlx/backend/metal/device.cpp",
        "mlx/mlx/backend/metal/device_info.cpp",
        "mlx/mlx/backend/metal/distributed.cpp",
        "mlx/mlx/backend/metal/eval.cpp",
        "mlx/mlx/backend/metal/event.cpp",
        "mlx/mlx/backend/metal/fence.cpp",
        "mlx/mlx/backend/metal/fft.cpp",
        "mlx/mlx/backend/metal/hadamard.cpp",
        "mlx/mlx/backend/metal/indexing.cpp",
        "mlx/mlx/backend/metal/jit_kernels.cpp",
        "mlx/mlx/backend/metal/logsumexp.cpp",
        "mlx/mlx/backend/metal/matmul.cpp",
        "mlx/mlx/backend/metal/metal.cpp",
        "mlx/mlx/backend/metal/normalization.cpp",
        "mlx/mlx/backend/metal/primitives.cpp",
        "mlx/mlx/backend/metal/quantized.cpp",
        "mlx/mlx/backend/metal/reduce.cpp",
        "mlx/mlx/backend/metal/resident.cpp",
        "mlx/mlx/backend/metal/rope.cpp",
        "mlx/mlx/backend/metal/scaled_dot_product_attention.cpp",
        "mlx/mlx/backend/metal/scan.cpp",
        "mlx/mlx/backend/metal/slicing.cpp",
        "mlx/mlx/backend/metal/softmax.cpp",
        "mlx/mlx/backend/metal/sort.cpp",
        "mlx/mlx/backend/metal/ternary.cpp",
        "mlx/mlx/backend/metal/unary.cpp",
        "mlx/mlx/backend/metal/utils.cpp",
        "mlx/mlx/backend/metal/kernels",  // Exclude kernels directory
        "mlx/mlx/backend/metal/jit",  // Exclude jit directory

        "mlx/mlx/backend/gpu",  // Exclude GPU backend on Linux, use no_gpu instead
        "mlx/mlx/backend/no_cpu",  // Exclude no_cpu backend on Linux, use cpu instead
        "mlx/mlx/backend/cpu/gemms/bnns.cpp",  // macOS Accelerate version
        "mlx-conditional",
        "mlx-c/mlx/c/metal.cpp",

        "mlx-c/mlx/c/fast.cpp",  // Exclude on Linux - calls metal_kernel unconditionally
    ]

    let cxxSettings: [CXXSetting] = []

    let linkerSettings: [LinkerSetting] = [
        .linkedLibrary("gfortran", .when(platforms: [.linux])),
        .linkedLibrary("blas", .when(platforms: [.linux])),
        .linkedLibrary("lapack", .when(platforms: [.linux])),
        .linkedLibrary("openblas", .when(platforms: [.linux])),
    ]

    let mlxSwiftExcludes: [String] = [
        "GPU+Metal.swift",
        "MLXArray+Metal.swift",
        "MLXFast.swift",
        "MLXFastKernel.swift",
    ]
#else
    let platformExcludes: [String] = [
        "mlx/mlx/backend/cpu/compiled.cpp",

        // opt-out of these backends (using metal)
        "mlx/mlx/backend/no_gpu",
        "mlx/mlx/backend/no_cpu",
        "mlx/mlx/backend/metal/no_metal.cpp",

        // bnns instead of simd (accelerate)
        "mlx/mlx/backend/cpu/gemms/simd_fp16.cpp",
        "mlx/mlx/backend/cpu/gemms/simd_bf16.cpp",
    ]

    let cxxSettings: [CXXSetting] = [
        .headerSearchPath("metal-cpp"),

        .define("MLX_USE_ACCELERATE"),
        .define("ACCELERATE_NEW_LAPACK"),
        .define("_METAL_"),
        .define("SWIFTPM_BUNDLE", to: "\"mlx-swift_Cmlx\""),
        .define("METAL_PATH", to: "\"default.metallib\""),
    ]

    let linkerSettings: [LinkerSetting] = [
        .linkedFramework("Foundation"),
        .linkedFramework("Metal"),
        .linkedFramework("Accelerate"),
    ]

    let mlxSwiftExcludes: [String] = []
#endif

let transformersSwiftSettings: [SwiftSetting] = [
    .enableExperimentalFeature("StrictConcurrency")
]

let mlxLMCommonExcludedFiles: [String] = [
    "README.md",
    "BatchEngine/BATCH_ENGINE.md",
    "BatchEngine/DSV4-OSAURUS-HOOKUP.md",
    "BatchEngine/FORK-SYNC-PROCESS.md",
    "BatchEngine/GEMMA4-SLIDING-WINDOW-CRASH.md",
    "BatchEngine/JANGTQ-RUNTIME-PATCH-GUIDE.md",
    "BatchEngine/KV-SIZING-CONTRACT.md",
    "BatchEngine/LOW-SPEC-HOST-GUIDANCE.md",
    "BatchEngine/MCDC-COVERAGE-STRATEGY.md",
    "BatchEngine/MEDIA-MODEL-MATRIX.md",
    "BatchEngine/MODEL-LOADING-STATUS-2026-05-01.md",
    "BatchEngine/OMNI-OSAURUS-HOOKUP.md",
    "BatchEngine/OMNI-VOICE-INTEGRATION.md",
    "BatchEngine/OSAURUS-API-SURFACE.md",
    "BatchEngine/OSAURUS-INTEGRATION-2026-05-01.md",
    "BatchEngine/OSAURUS-INTEGRATION.md",
    "BatchEngine/OSAURUS-PRODUCTION-REFERENCE-2026-05-01.md",
    "BatchEngine/OSAURUS-RELEASE-HANDOFF.md",
    "BatchEngine/OSAURUS-RUNTIME-HANDOFF-2026-05-06.md",
    "BatchEngine/OSAURUS-TEAM-BUILD-GUIDE.md",
    "BatchEngine/PARAKEET-RADIO-INTEGRATION.md",
    "BatchEngine/RALPH-EDGE-CASE-STATE.md",
    "BatchEngine/RALPH-EDGE-TASK.md",
    "BatchEngine/RALPH-TASK-HARMONY.md",
    "BatchEngine/REASONING-STREAM-EVENT.md",
    "BatchEngine/STAGE-1B4-DESIGN-2026-05-02.md",
    "BatchEngine/STOP-SEQUENCES-CONTRACT.md",
    "BatchEngine/TPAE-2026-04-20-TRIAGE.md",
    "Cache/COLD-WEIGHT-TIER-DESIGN.md",
    "Cache/JANGPRESS-AGENTS.md",
    "Cache/JANGPRESS-DEEP-TRACE.md",
    "Cache/JANGPRESS-INTEGRATION.md",
    "Cache/JANGPRESS-MEASUREMENTS.md",
    "Cache/JANGPRESS-PER-MODEL-RESULTS.md",
    "Cache/JANGPRESS-PRODUCTION.md",
    "Cache/JANGPRESS-STATUS.md",
    "Cache/JANGPRESS-USAGE.md",
    "Cache/JANGPRESS-VMLX-SWIFT-LM-INTEGRATION-2026-05-02.md",
    "ChatTemplates/DSV4Minimal.jinja",
    "ChatTemplates/Gemma4Minimal.jinja",
    "ChatTemplates/Gemma4WithTools.jinja",
    "ChatTemplates/MiniMaxM2Minimal.jinja",
    "ChatTemplates/NemotronMinimal.jinja",
    "ChatTemplates/swift-jinja-patches/0001-lexer-curly-ambiguity.patch",
    "ChatTemplates/swift-jinja-patches/0002-runtime-dict-iter-and-select-expression.patch",
    "ChatTemplates/swift-jinja-patches/README.md",
    "SpecDec/DDTREE-DESIGN.md",
    "SpecDec/OSAURUS-SPECDEC.md",
    "TOOL-CALL-STRUCTURED-CONTRACT.md",
]

let mlxDistributedCoreExcludedFiles: [String] = [
    "README.md",
    "DISTRIBUTED-INFERENCE-ROADMAP.md",
    "EXO-ARCHITECTURE-REVIEW.md",
    "JACCL-RDMA-DISCOVERY-BRINGUP.md",
    "OSAURUS-DISTRIBUTED-INTEGRATION.md",
    "TWO-M5-BRINGUP.md",
]

let cmlx = Target.target(
    name: "Cmlx",
    path: "Source/Cmlx",
    exclude: platformExcludes + [
        // vendor docs
        "vendor-README.md",

        // example code + mlx-c distributed
        "mlx-c/examples",
        "mlx-c/mlx/c/distributed.cpp",
        "mlx-c/mlx/c/distributed_group.cpp",

        // vendored library, include header only
        "json",

        // vendored library
        "fmt/test",
        "fmt/doc",
        "fmt/support",
        "fmt/src/os.cc",
        "fmt/src/fmt.cc",

        // these are selected conditionally
        "mlx/mlx/backend/no_cpu/compiled.cpp",

        // mlx files that are not part of the build
        "mlx/ACKNOWLEDGMENTS.md",
        "mlx/CMakeLists.txt",
        "mlx/CODE_OF_CONDUCT.md",
        "mlx/CONTRIBUTING.md",
        "mlx/LICENSE",
        "mlx/MANIFEST.in",
        "mlx/README.md",
        "mlx/benchmarks",
        "mlx/cmake",
        "mlx/docs",
        "mlx/examples",
        "mlx/mlx.pc.in",
        "mlx/pyproject.toml",
        "mlx/python",
        "mlx/setup.py",
        "mlx/tests",

        // special handling for cuda -- we need to keep one file:
        // mlx/mlx/backend/cuda/no_cuda.cpp

        "mlx/mlx/backend/cuda/allocator.cpp",
        "mlx/mlx/backend/cuda/compiled.cpp",
        "mlx/mlx/backend/cuda/conv.cpp",
        "mlx/mlx/backend/cuda/cublas_utils.cpp",
        "mlx/mlx/backend/cuda/cudnn_utils.cpp",
        "mlx/mlx/backend/cuda/custom_kernel.cpp",
        "mlx/mlx/backend/cuda/delayload.cpp",
        "mlx/mlx/backend/cuda/device_info.cpp",
        "mlx/mlx/backend/cuda/device.cpp",
        "mlx/mlx/backend/cuda/eval.cpp",
        "mlx/mlx/backend/cuda/fence.cpp",
        "mlx/mlx/backend/cuda/indexing.cpp",
        "mlx/mlx/backend/cuda/jit_module.cpp",
        "mlx/mlx/backend/cuda/load.cpp",
        "mlx/mlx/backend/cuda/matmul.cpp",
        "mlx/mlx/backend/cuda/primitives.cpp",
        "mlx/mlx/backend/cuda/scaled_dot_product_attention.cpp",
        "mlx/mlx/backend/cuda/slicing.cpp",
        "mlx/mlx/backend/cuda/utils.cpp",
        "mlx/mlx/backend/cuda/worker.cpp",

        "mlx/mlx/backend/cuda/binary",
        "mlx/mlx/backend/cuda/conv",
        "mlx/mlx/backend/cuda/copy",
        "mlx/mlx/backend/cuda/device",
        "mlx/mlx/backend/cuda/gemms",
        "mlx/mlx/backend/cuda/quantized",
        "mlx/mlx/backend/cuda/reduce",
        "mlx/mlx/backend/cuda/steel",
        "mlx/mlx/backend/cuda/unary",

        // build variants (we are opting _out_ of these)
        "mlx/mlx/io/no_safetensors.cpp",
        "mlx/mlx/io/gguf.cpp",
        "mlx/mlx/io/gguf_quants.cpp",

        // see PrepareMetalShaders -- don't build the kernels in place
        "mlx/mlx/backend/metal/kernels",
        "mlx/mlx/backend/metal/nojit_kernels.cpp",

        // do not build distributed support (yet)
        "mlx/mlx/distributed/mpi/mpi.cpp",
        "mlx/mlx/distributed/ring/ring.cpp",
        "mlx/mlx/distributed/nccl/nccl.cpp",
        "mlx/mlx/distributed/nccl/nccl_stub",
        "mlx/mlx/distributed/jaccl/jaccl.cpp",
        "mlx/mlx/distributed/jaccl/mesh.cpp",
        "mlx/mlx/distributed/jaccl/ring.cpp",
        "mlx/mlx/distributed/jaccl/utils.cpp",
    ],
    cSettings: [
        .headerSearchPath("mlx"),
        .headerSearchPath("mlx-c"),
    ],
    cxxSettings: cxxSettings + [
        .headerSearchPath("mlx"),
        .headerSearchPath("mlx-c"),
        .headerSearchPath("json/single_include/nlohmann"),
        .headerSearchPath("fmt/include"),
        .define("MLX_VERSION", to: "\"0.31.1\""),
    ],
    linkerSettings: linkerSettings
)

let package = Package(
    name: "mlx-swift",

    platforms: [
        .macOS("14.0"),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],

    products: [
        // main targets
        .library(name: "MLX", targets: ["MLX"]),
        .library(name: "MLXRandom", targets: ["MLXRandom"]),
        .library(name: "MLXNN", targets: ["MLXNN"]),
        .library(name: "MLXOptimizers", targets: ["MLXOptimizers"]),
        .library(name: "MLXFFT", targets: ["MLXFFT"]),
        .library(name: "MLXLinalg", targets: ["MLXLinalg"]),
        .library(name: "MLXFast", targets: ["MLXFast"]),
        .library(name: "VMLXJinja", targets: ["VMLXJinja"]),
        .library(name: "VMLXHub", targets: ["VMLXHub"]),
        .library(name: "VMLXTokenizers", targets: ["VMLXTokenizers"]),
        .library(name: "VMLXTransformers", targets: ["VMLXTokenizers", "VMLXGeneration", "VMLXModels"]),
        .library(name: "MLXLMCommon", targets: ["MLXLMCommon"]),
        .library(name: "MLXLLM", targets: ["MLXLLM"]),
        .library(name: "MLXVLM", targets: ["MLXVLM"]),
        .library(name: "MLXEmbedders", targets: ["MLXEmbedders"]),
        .library(name: "MLXHuggingFace", targets: ["MLXHuggingFace"]),
        .library(name: "BenchmarkHelpers", targets: ["BenchmarkHelpers"]),
        .library(name: "IntegrationTestHelpers", targets: ["IntegrationTestHelpers"]),
        .library(name: "MLXDistributedCore", targets: ["MLXDistributedCore"]),
        .library(name: "MLXDistributedTransport", targets: ["MLXDistributedTransport"]),
        .library(name: "MLXDistributedJACCL", targets: ["MLXDistributedJACCL"]),
        .library(name: "MLXDistributedTP", targets: ["MLXDistributedTP"]),
        .library(name: "MLXPress", targets: ["MLXPress"]),
        .library(name: "VMLX", targets: ["VMLX"]),
        .executable(name: "RunBench", targets: ["RunBench"]),
        .executable(name: "ANEProbe", targets: ["ANEProbe"]),
        .executable(name: "OmniAudioLatencyBench", targets: ["OmniAudioLatencyBench"]),
        .executable(name: "OmniAudioChunkStabilityBench", targets: ["OmniAudioChunkStabilityBench"]),
        .executable(name: "mlxpress", targets: ["MLXPressCLI"]),
        .executable(name: "mlxpress-selfcheck", targets: ["MLXPressSelfCheck"]),
    ],
    dependencies: [
        // for Complex type
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0"),
        .package(url: "https://github.com/swiftlang/swift-syntax.git", from: "600.0.0-latest"),
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.65.0"),
        .package(url: "https://github.com/apple/swift-nio-ssl.git", from: "2.27.0"),
        .package(url: "https://github.com/apple/swift-nio-http2.git", from: "1.34.0"),
        .package(url: "https://github.com/apple/swift-certificates.git", from: "1.5.0"),
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.5.0"),
        .package(url: "https://github.com/apple/swift-crypto.git", "3.0.0"..<"5.0.0"),
        .package(url: "https://github.com/swift-server/async-http-client.git", from: "1.33.0"),
        .package(url: "https://github.com/ibireme/yyjson.git", from: "0.12.0"),
    ],
    targets: [
        cmlx,
        .testTarget(
            name: "CmlxTests",
            dependencies: ["Cmlx"]
        ),

        .target(
            name: "MLX",
            dependencies: [
                "Cmlx",
                .product(name: "Numerics", package: "swift-numerics"),
            ],
            path: "Source/MLX",
            exclude: mlxSwiftExcludes,
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXRandom",
            dependencies: ["MLX"],
            path: "Source/MLXRandom",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXFast",
            dependencies: ["MLX", "Cmlx"],
            path: "Source/MLXFast",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXNN",
            dependencies: ["MLX"],
            path: "Source/MLXNN",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXOptimizers",
            dependencies: ["MLX", "MLXNN"],
            path: "Source/MLXOptimizers",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXFFT",
            dependencies: ["MLX"],
            path: "Source/MLXFFT",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .target(
            name: "MLXLinalg",
            dependencies: ["MLX"],
            path: "Source/MLXLinalg",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),

        .target(
            name: "VMLXJinja",
            dependencies: [
                .product(name: "OrderedCollections", package: "swift-collections"),
            ],
            path: "Vendors/Jinja/Sources/Jinja"
        ),
        .target(
            name: "VMLXEventSource",
            dependencies: [
                .product(name: "AsyncHTTPClient", package: "async-http-client"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "Vendors/EventSource/Sources/EventSource"
        ),
        .target(
            name: "VMLXHuggingFace",
            dependencies: [
                "VMLXEventSource",
                .product(name: "Crypto", package: "swift-crypto"),
            ],
            path: "Vendors/swift-huggingface/Sources/HuggingFace"
        ),
        .target(
            name: "VMLXGeneration",
            dependencies: ["VMLXTokenizers"],
            path: "Vendors/swift-transformers/Sources/Generation"
        ),
        .target(
            name: "VMLXHub",
            dependencies: [
                "VMLXJinja",
                "VMLXHuggingFace",
                .product(name: "OrderedCollections", package: "swift-collections"),
                .product(name: "Crypto", package: "swift-crypto"),
                .product(name: "yyjson", package: "yyjson"),
            ],
            path: "Vendors/swift-transformers/Sources/Hub",
            resources: [.process("Resources")],
            swiftSettings: transformersSwiftSettings
        ),
        .target(
            name: "VMLXModels",
            dependencies: ["VMLXTokenizers", "VMLXGeneration"],
            path: "Vendors/swift-transformers/Sources/Models"
        ),
        .target(
            name: "VMLXTokenizers",
            dependencies: ["VMLXHub", "VMLXJinja"],
            path: "Vendors/swift-transformers/Sources/Tokenizers"
        ),

        .target(
            name: "MLXLMCommon",
            dependencies: ["MLX", "MLXNN", "MLXOptimizers", "MLXRandom"],
            path: "Libraries/MLXLMCommon",
            exclude: mlxLMCommonExcludedFiles
        ),
        .target(
            name: "MLXLLM",
            dependencies: ["MLXLMCommon", "MLX", "MLXNN", "MLXOptimizers"],
            path: "Libraries/MLXLLM",
            exclude: ["README.md", "Models/DSV4-PORT-STATUS.md"]
        ),
        .target(
            name: "MLXVLM",
            dependencies: ["MLXLMCommon", "MLXLLM", "MLX", "MLXNN", "MLXOptimizers"],
            path: "Libraries/MLXVLM",
            exclude: ["README.md"]
        ),
        .target(
            name: "MLXEmbedders",
            dependencies: ["MLX", "MLXNN", "MLXLMCommon"],
            path: "Libraries/MLXEmbedders",
            exclude: ["README.md"]
        ),
        .target(
            name: "MLXDistributedCore",
            dependencies: [],
            path: "Libraries/MLXDistributedCore",
            exclude: mlxDistributedCoreExcludedFiles
        ),
        .target(
            name: "MLXDistributedTransport",
            dependencies: [
                "MLXDistributedCore",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOSSL", package: "swift-nio-ssl"),
                .product(name: "NIOHTTP2", package: "swift-nio-http2"),
                .product(name: "X509", package: "swift-certificates"),
                .product(name: "Crypto", package: "swift-crypto"),
            ],
            path: "Libraries/MLXDistributedTransport",
            exclude: ["README.md"]
        ),
        .target(
            name: "MLXDistributedJACCL",
            dependencies: ["MLXDistributedCore", "MLX"],
            path: "Libraries/MLXDistributedJACCL",
            exclude: ["README.md"]
        ),
        .target(
            name: "CmlxDistributedShim",
            dependencies: [],
            path: "Libraries/CmlxDistributedShim",
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("../../Source/Cmlx/mlx-c"),
            ],
            cxxSettings: [
                .headerSearchPath("../../Source/Cmlx/mlx-c"),
                .headerSearchPath("../../Source/Cmlx/mlx"),
                .headerSearchPath("../../Source/Cmlx/json/single_include/nlohmann"),
                .headerSearchPath("../../Source/Cmlx/fmt/include"),
            ]
        ),
        .target(
            name: "CmlxGraphShim",
            dependencies: [],
            path: "Libraries/CmlxGraphShim",
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("../../Source/Cmlx/mlx-c"),
            ],
            cxxSettings: [
                .headerSearchPath("../../Source/Cmlx/mlx-c"),
                .headerSearchPath("../../Source/Cmlx/mlx"),
                .headerSearchPath("../../Source/Cmlx/fmt/include"),
            ]
        ),
        .target(
            name: "MLXDistributedTP",
            dependencies: [
                "MLXDistributedCore",
                "MLXDistributedJACCL",
                "CmlxDistributedShim",
                "MLX",
                "MLXNN",
            ],
            path: "Libraries/MLXDistributedTP",
            exclude: ["README.md"]
        ),
        .target(
            name: "BenchmarkHelpers",
            dependencies: ["MLXLMCommon", "MLXLLM", "MLXVLM", "MLXEmbedders", "MLX"],
            path: "Libraries/BenchmarkHelpers"
        ),
        .target(
            name: "IntegrationTestHelpers",
            dependencies: ["MLXLMCommon", "MLXLLM", "MLXVLM", "MLXEmbedders", "MLX"],
            path: "Libraries/IntegrationTestHelpers",
            exclude: ["README.md"]
        ),
        .macro(
            name: "MLXHuggingFaceMacros",
            dependencies: [
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
            ],
            path: "Libraries/MLXHuggingFaceMacros"
        ),
        .target(
            name: "MLXHuggingFace",
            dependencies: ["MLXHuggingFaceMacros", "MLXLMCommon"],
            path: "Libraries/MLXHuggingFace"
        ),
        .target(
            name: "MLXPress",
            dependencies: [
                "MLXLLM",
                "MLXLMCommon",
                "MLXHuggingFace",
                "VMLXTokenizers",
            ],
            path: "Sources/MLXPress"
        ),
        .target(
            name: "VMLX",
            dependencies: [
                "MLX",
                "MLXRandom",
                "MLXNN",
                "MLXOptimizers",
                "MLXFFT",
                "MLXLinalg",
                "MLXFast",
                "VMLXJinja",
                "VMLXHub",
                "VMLXTokenizers",
                "VMLXGeneration",
                "VMLXModels",
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                "MLXHuggingFace",
                "MLXDistributedCore",
                "MLXDistributedTransport",
                "MLXDistributedJACCL",
                "MLXDistributedTP",
                "MLXPress",
            ],
            path: "Libraries/VMLX",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "MLXPressCLI",
            dependencies: ["MLXPress", "MLXLMCommon"],
            path: "Sources/MLXPressCLI"
        ),
        .executableTarget(
            name: "MLXPressSelfCheck",
            dependencies: ["MLXPress"],
            path: "Sources/MLXPressSelfCheck"
        ),
        .executableTarget(
            name: "CompileBench",
            dependencies: ["MLX", "MLXNN", "MLXRandom"],
            path: "CompileBench"
        ),
        .executableTarget(
            name: "TPRankWorker",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXHuggingFace",
                "MLXDistributedTP",
                "MLXDistributedJACCL",
                "MLX",
                "MLXNN",
                "VMLXTokenizers",
            ],
            path: "tools/TPRankWorker"
        ),
        .executableTarget(
            name: "ANEProbe",
            dependencies: ["MLXLMCommon"],
            path: "tools/ANEProbe"
        ),
        .executableTarget(
            name: "OmniAudioLatencyBench",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXHuggingFace",
                "MLX",
                "VMLXTokenizers",
            ],
            path: "tools/OmniAudioLatencyBench"
        ),
        .executableTarget(
            name: "OmniAudioChunkStabilityBench",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXHuggingFace",
                "MLX",
                "VMLXTokenizers",
            ],
            path: "tools/OmniAudioChunkStabilityBench"
        ),
        .executableTarget(
            name: "RunBench",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXHuggingFace",
                "CmlxGraphShim",
                "MLX",
                "VMLXJinja",
                "VMLXTokenizers",
                "VMLXGeneration",
                "VMLXModels",
            ],
            path: "RunBench",
            exclude: ["coherency-matrix.sh", "test_slice.swift.bak"]
        ),

        .testTarget(
            name: "MLXTests",
            dependencies: [
                "MLX", "MLXNN", "MLXOptimizers",
            ],
            path: "Tests/MLXTests"
        ),
        .testTarget(
            name: "MLXLMTests",
            dependencies: [
                "MLX",
                "MLXNN",
                "MLXOptimizers",
                "VMLXJinja",
                "VMLXTokenizers",
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                "MLXHuggingFace",
                "BenchmarkHelpers",
            ],
            path: "Tests/MLXLMTests",
            exclude: [
                "README.md",
            ],
            resources: [
                .process("Resources/1080p_30.mov"),
                .process("Resources/audio_only.mov"),
            ]
        ),
        .testTarget(
            name: "MLXPressPolicyTests",
            path: "Tests/MLXPressPolicyTests",
            sources: ["MLXPressLowRamPolicySourceTests.swift"]
        ),
        .testTarget(
            name: "MLXLMCommonFocusedTests",
            dependencies: ["MLX", "MLXLMCommon", "MLXLLM", "MLXVLM", "VMLXJinja", "VMLX"],
            path: "Tests/MLXLMCommonFocusedTests",
            sources: [
                "DeepseekV4ChatTemplateFallbackFocusedTests.swift",
                "DeepseekV4Step37RuntimeContractsTests.swift",
                "DSMLInlineJSONToolFallbackFocusedTests.swift",
                "DSMLToolCallParserFocusedTests.swift",
                "FocusedMLXTestSupport.swift",
                "BatchEngineGrowingChatCacheSourceTests.swift",
                "CacheCoordinatorTopologyFocusedTests.swift",
                "VMLXUmbrellaProductTests.swift",
                "ZayaConfigDecodeFocusedTests.swift",
                "VLShapeGuardFocusedTests.swift",
                "JANGTQStreamingExpertDescriptorTests.swift",
                "JANGTQHadamardShuffleTests.swift",
                "MiniMaxJANGTQResidentExpertTests.swift",
                "JangPressMachCacheTests.swift",
                "JangPressPrestackerCleanupTests.swift",
                "DeepseekV4IndexerCausalTopKTests.swift",
                "DeepseekV4ReasoningPolicyTests.swift",
                "Gemma4ZyphraToolParserFocusedTests.swift",
                "Gemma4ThoughtChannelParserFocusedTests.swift",
                "Gemma3nTextSanitizeFocusedTests.swift",
                "MediaCachePlaceholderTests.swift",
                "NemotronHOmniPreEncodedAudioTests.swift",
                "MTPRuntimeFocusedTests.swift",
                "MLXPressCLISourceContractsTests.swift",
                "VMLXServerRuntimeSettingsTests.swift",
                "DSV4AgenticToolSourceTests.swift",
                "NoHiddenReasoningCloseBiasFocusedTests.swift",
                "TokenizerAddedTokenRegexFocusedTests.swift",
            ]
        ),
        .testTarget(
            name: "MLXLMCommonToolParserFocusedTests",
            dependencies: ["MLXLMCommon", "VMLXJinja"],
            path: "Tests/MLXLMCommonToolParserFocusedTests"
        ),

        // ------
        // Example programs

        .executableTarget(
            name: "Example1",
            dependencies: ["MLX"],
            path: "Source/Examples",
            exclude: ["Tutorial.swift", "CustomFunctionExample.swift", "CustomFunctionExampleSimple.swift"],
            sources: ["Example1.swift"]
        ),
        .executableTarget(
            name: "Tutorial",
            dependencies: ["MLX"],
            path: "Source/Examples",
            exclude: ["Example1.swift", "CustomFunctionExample.swift", "CustomFunctionExampleSimple.swift"],
            sources: ["Tutorial.swift"]
        ),
        .executableTarget(
            name: "CustomFunctionExample",
            dependencies: ["MLX"],
            path: "Source/Examples",
            exclude: ["Example1.swift", "Tutorial.swift", "CustomFunctionExampleSimple.swift"],
            sources: ["CustomFunctionExample.swift"]
        ),
        .executableTarget(
            name: "CustomFunctionExampleSimple",
            dependencies: ["MLX"],
            path: "Source/Examples",
            exclude: ["Example1.swift", "Tutorial.swift", "CustomFunctionExample.swift"],
            sources: ["CustomFunctionExampleSimple.swift"]
        ),
    ],
    cxxLanguageStandard: .gnucxx20
)

if Context.environment["MLX_SWIFT_BUILD_DOC"] == "1"
    || Context.environment["SPI_GENERATE_DOCS"] == "1"
{
    // docc builder
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0")
    )
}
