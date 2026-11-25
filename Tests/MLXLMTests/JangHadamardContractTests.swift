// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import MLXLMCommon

/// Small, internally consistent copy of the converter's three JSON surfaces.
/// No model or MLX evaluation is needed for these metadata regressions.
struct JangHadamardFixture {
    static let forward = "projection"
    static let inverse = "embedding"
    static let width = 512
    static let signs: [Float] = (0 ..< width).map { $0 % 7 < 3 ? -1 : 1 }

    var config: [String: Any]
    var jang: [String: Any]
    var sidecar: [String: Any]

    init(packed: Bool = false) {
        let hadamard: [String: Any] = [
            "contract": "prism.hadamard.v1", "sidecar": "hadamard.json",
            "block_size": Self.width,
            "transform": "normalized-sylvester-walsh-hadamard",
            "axis": "input-last-dimension", "sign_mode": "explicit",
            "signs_dtype": "float32", "signs_tensor_suffix": "signs",
            "compute_dtype": "float32", "gdn_v_grouped": true,
            "forward_modules": [Self.forward], "inverse_modules": [Self.inverse],
        ]
        var manifest: [String: Any] = [:]
        for path in [Self.forward, Self.inverse] {
            var entry: [String: Any] = [
                "bits": 2, "group_size": 128, "mode": "affine",
                "hadamard": [
                    "block_size": Self.width,
                    "direction": path == Self.forward ? "forward" : "inverse",
                    "signs_tensor": "\(path).signs",
                ],
            ]
            if packed {
                entry["storage"] = "ternary_packed_26b"
                entry["runtime_bits"] = 2
            } else {
                entry["storage_bits"] = 2
            }
            manifest[path] = entry
        }
        var quantization: [String: Any] = [
            "bits": 2, "block_size": 128, "mode": "affine", "bit_widths_used": [2],
            "tensor_quantization_manifest_schema": 2,
            "tensor_quantization_manifest_count": manifest.count,
            "tensor_quantization_manifest": manifest,
        ]
        quantization["ternary_packed_runtime_expansion"] =
            packed
            ? [
                "storage": "ternary_packed_26b", "runtime_bits": 2, "group_size": 128,
                "lossless": true, "scales_unchanged": true,
                "biases": "materialized as -scales at load",
            ] : NSNull()
        config = [
            "model_type": "qwen3_5", "hidden_size": Self.width,
            "quantization": ["bits": 2, "group_size": 128, "mode": "affine"],
            "hadamard": hadamard,
        ]
        jang = [
            "format": "jang", "format_version": "2.0", "weight_format": "affine",
            "quantization": quantization, "hadamard": hadamard,
            "runtime": [
                "requires_hadamard_activation_transform": true,
                "requires_jang_ternary_packed_expansion": packed,
                "requires_jang_affine1_expansion": false,
            ],
        ]
        sidecar = [
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": Self.width,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.gdn_v_grouped": true,
            "prism.hadamard.weight_names": ["\(Self.forward).weight"],
            "prism.hadamard.inverse_weight_names": ["\(Self.inverse).weight"],
            "prism.hadamard.sign_widths": [Self.width],
            "prism.hadamard.sign_values": Self.signs,
        ]
    }

    func write(to directory: URL) throws {
        for (name, json) in [
            ("config.json", config), ("jang_config.json", jang), ("hadamard.json", sidecar),
        ] {
            try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys])
                .write(to: directory.appendingPathComponent(name))
        }
    }

    func withDirectory<T>(_ body: (URL) throws -> T) throws -> T {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("jang-bonsai2-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try write(to: directory)
        return try body(directory)
    }
}

@Suite("JANG Hadamard and packed ternary contracts")
struct JangHadamardContractTests {
    @Test("original Prism module declarations agree with the signed transform sidecar")
    func prismPublishedBundle() throws {
        var fixture = JangHadamardFixture()
        fixture.jang = [:]
        fixture.config = [
            "model_type": JangLoader.prismHadamardModelType,
            "schema_version": 2,
            "base_model_type": "qwen3_5",
            "tensor_namespace": "mlx-vlm-qwen3_5",
            "gdn_activation_layout": "grouped",
            "hadamard_config": "hadamard.json",
            "components": ["text": true, "vision": true, "mtp": false],
            "quantization": ["bits": 2, "group_size": 128, "mode": "affine"],
            "modules": [
                ["path": "model.projection", "block": 512, "embedding": false, "dtype": "float16"],
                ["path": "model.embed_tokens", "block": 512, "embedding": true, "dtype": "float16"],
            ],
        ]
        fixture.sidecar["prism.hadamard.weight_names"] = ["language_model.model.projection.weight"]
        fixture.sidecar["prism.hadamard.inverse_weight_names"] =
            ["language_model.model.embed_tokens.weight"]
        try fixture.withDirectory { directory in
            let loaded = try JangLoader.loadHadamardRuntimeContract(at: directory)
            let contract = try #require(loaded)
            #expect(contract.supportsPrismV2MTP)
            #expect(contract.forward == ["language_model.model.projection"])
            #expect(contract.inverse == ["language_model.model.embed_tokens"])
            #expect(contract.sidecarSigns[512] == JangHadamardFixture.signs)
            var mismatched = fixture.sidecar
            mismatched["prism.hadamard.weight_names"] = ["language_model.model.other.weight"]
            try JSONSerialization.data(withJSONObject: mismatched)
                .write(to: directory.appendingPathComponent("hadamard.json"))
            #expect(throws: JangLoaderError.self) {
                _ = try JangLoader.loadHadamardRuntimeContract(at: directory)
            }
        }
    }

    @Test("Prism draft attachment requires a declared Qwen3.5 Hadamard repack", arguments: [false, true])
    func prismDraftRepack(packed: Bool) throws {
        var fixture = JangHadamardFixture(packed: packed)
        var runtime = try #require(fixture.jang["runtime"] as? [String: Any])
        runtime["prism_mtp_v2_sidecar"] = true
        fixture.jang["runtime"] = runtime
        fixture.jang["source_model"] = ["model_type": JangLoader.prismHadamardModelType]

        try fixture.withDirectory { directory in
            let contract = try JangLoader.loadHadamardRuntimeContract(at: directory)
            let attached = try #require(contract)
            #expect(attached.supportsPrismV2MTP)
            var incompatible = fixture.jang
            incompatible["source_model"] = ["model_type": "qwen3_5"]
            try JSONSerialization.data(withJSONObject: incompatible)
                .write(to: directory.appendingPathComponent("jang_config.json"))
            #expect(throws: JangLoaderError.self) {
                _ = try JangLoader.loadHadamardRuntimeContract(at: directory)
            }
        }
    }

    @Test("converter contracts select exact module coverage", arguments: [false, true])
    func converterContract(packed: Bool) throws {
        try JangHadamardFixture(packed: packed).withDirectory { directory in
            let loaded = try JangLoader.loadHadamardRuntimeContract(at: directory)
            let contract = try #require(loaded)
            #expect(contract.blockSize == 512)
            #expect(contract.computeDType == .float32)
            #expect(contract.forward == [JangHadamardFixture.forward])
            #expect(contract.inverse == [JangHadamardFixture.inverse])
            #expect(contract.sidecarSigns[512] == JangHadamardFixture.signs)
            let storage = try JangLoader.loadTernaryPackedRuntimeContract(at: directory)
            #expect(packed ? storage?.modulePaths == contract.modulePaths : storage == nil)
            let affine1 = try JangLoader.loadAffine1RuntimeContract(at: directory)
            #expect(affine1 == nil)
        }
    }

    @Test("sidecar owner fallback and embedded JANG declaration are honored")
    func fallbackOwners() throws {
        var fixture = JangHadamardFixture()
        fixture.config.removeValue(forKey: "hadamard")
        try fixture.withDirectory { directory in
            let sidecarContract = try JangLoader.loadHadamardRuntimeContract(at: directory)
            #expect(sidecarContract != nil)
            var config = fixture.config
            config["jang_config"] = fixture.jang
            try JSONSerialization.data(withJSONObject: config)
                .write(to: directory.appendingPathComponent("config.json"))
            try FileManager.default.removeItem(
                at: directory.appendingPathComponent("jang_config.json"))
            let embeddedContract = try JangLoader.loadHadamardRuntimeContract(at: directory)
            #expect(embeddedContract != nil)
        }
    }

    @Test("ordinary affine bundle has no activation or packed-storage contract")
    func ordinaryBundle() throws {
        var fixture = JangHadamardFixture()
        fixture.config.removeValue(forKey: "hadamard")
        fixture.jang = ["format": "jang", "quantization": ["bits": 2, "group_size": 128]]
        try fixture.withDirectory { directory in
            let hadamard = try JangLoader.loadHadamardRuntimeContract(at: directory)
            let packed = try JangLoader.loadTernaryPackedRuntimeContract(at: directory)
            #expect(hadamard == nil)
            #expect(packed == nil)
        }
    }

    @Test(
        "both storage declarations retain the bundle's native reasoning vocabulary",
        arguments: [false, true])
    func nativeReasoningDeclaration(packed: Bool) throws {
        var fixture = JangHadamardFixture(packed: packed)
        // Exact top-level schema used by both private Bonsai2 bundles. This
        // tests capability transport, not a substitute for rendering the
        // complete native template or generating coherent model output.
        fixture.jang["reasoning"] = [
            "supported": true, "parser": "qwen3", "default": "on",
            "enable_kwarg": "enable_thinking",
            "supported_reasoning_efforts": ["low", "medium", "xhigh"],
            "default_reasoning_effort": "xhigh",
            "reasoning_effort_transport": "chat_template_kwarg",
            "preserve_thinking_supported": true, "preserve_thinking_default": true,
            "preserve_thinking_transport": "chat_template_kwarg",
        ]
        fixture.jang["tools"] = [
            "supported": true, "parser": "qwen3_coder", "dialect": "xml_function",
        ]
        try fixture.withDirectory { directory in
            let template = """
                {%- if enable_thinking is undefined or enable_thinking is true %}
                    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
                    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
                        {{- raise_exception('Unexpected reasoning effort') }}
                    {%- endif %}
                {%- endif %}
                """
            try JSONSerialization.data(withJSONObject: ["chat_template": template])
                .write(to: directory.appendingPathComponent("tokenizer_config.json"))
            let capability = ReasoningCapability.forModel(at: directory)
            #expect(capability.source == .declared)
            #expect(capability.efforts == ["low", "medium", "xhigh"])
            #expect(capability.defaultLevel == 3)
            #expect(capability.levels == [0, 1, 2, 3])
            #expect(capability.templateKey == "reasoning_effort")
            let off = try #require(capability.applying(level: 0))
            #expect(off["enable_thinking"] as? Bool == false)
            #expect(off["reasoning_effort"] == nil)
            for (level, effort) in ["low", "medium", "xhigh"].enumerated() {
                let context = try #require(capability.applying(level: level + 1))
                #expect(context["enable_thinking"] as? Bool == true)
                #expect(context["reasoning_effort"] as? String == effort)
            }
            #expect(try JangLoader.loadHadamardRuntimeContract(at: directory) != nil)
        }
    }

    @Test(
        "malformed Hadamard metadata fails closed",
        arguments: [
            "missing-contract", "unknown-contract", "bad-transform", "bad-axis", "bad-sign-mode",
            "bad-sign-dtype", "bad-compute", "bad-block", "ungrouped-gdn", "duplicate-path",
            "overlap", "empty-paths", "owner-disagreement", "count", "manifest-direction",
            "manifest-sign-key", "manifest-coverage", "sidecar-missing", "sidecar-sign-value",
            "sidecar-sign-count", "sidecar-sign-width-overflow", "sidecar-module",
            "sidecar-version",
            "manifest-storage",
        ])
    func malformedHadamard(reason: String) throws {
        var fixture = JangHadamardFixture()
        var hadamard = fixture.config["hadamard"] as! [String: Any]
        var quantization = fixture.jang["quantization"] as! [String: Any]
        var manifest = quantization["tensor_quantization_manifest"] as! [String: Any]
        var entry = manifest[JangHadamardFixture.forward] as! [String: Any]
        var moduleHadamard = entry["hadamard"] as! [String: Any]
        switch reason {
        case "unknown-contract": hadamard["contract"] = "prism.hadamard.v2"
        case "bad-transform": hadamard["transform"] = "unnormalized"
        case "bad-axis": hadamard["axis"] = "output"
        case "bad-sign-mode": hadamard["sign_mode"] = "seed"
        case "bad-sign-dtype": hadamard["signs_dtype"] = "float16"
        case "bad-compute": hadamard["compute_dtype"] = "float16"
        case "bad-block": hadamard["block_size"] = 256
        case "ungrouped-gdn": hadamard["gdn_v_grouped"] = false
        case "duplicate-path": hadamard["forward_modules"] = ["projection", "projection"]
        case "overlap": hadamard["inverse_modules"] = ["projection"]
        case "empty-paths":
            hadamard["forward_modules"] = [String]()
            hadamard["inverse_modules"] = [String]()
        case "manifest-direction": moduleHadamard["direction"] = "inverse"
        case "manifest-sign-key": moduleHadamard["signs_tensor"] = "other.signs"
        case "manifest-storage": entry["storage"] = "unknown-packed"
        case "sidecar-sign-value":
            fixture.sidecar["prism.hadamard.sign_values"] = Array(repeating: 0, count: 512)
        case "sidecar-sign-count": fixture.sidecar["prism.hadamard.sign_values"] = [1]
        case "sidecar-sign-width-overflow":
            fixture.sidecar["prism.hadamard.sign_widths"] = [Int.max - 511, Int.max - 1023]
        case "sidecar-module": fixture.sidecar["prism.hadamard.weight_names"] = ["other.weight"]
        case "sidecar-version": fixture.sidecar["prism.hadamard.version"] = 2
        default: break
        }
        entry["hadamard"] = moduleHadamard
        manifest[JangHadamardFixture.forward] = entry
        if reason == "manifest-coverage" {
            manifest.removeValue(forKey: JangHadamardFixture.inverse)
        }
        quantization["tensor_quantization_manifest"] = manifest
        if reason == "count" { quantization["tensor_quantization_manifest_count"] = 99 }
        fixture.jang["quantization"] = quantization
        fixture.config["hadamard"] = hadamard
        if reason != "owner-disagreement" {
            fixture.jang["hadamard"] = hadamard
        } else {
            var other = hadamard
            other["block_size"] = 1024
            fixture.jang["hadamard"] = other
        }
        if reason == "missing-contract" {
            fixture.config.removeValue(forKey: "hadamard")
            fixture.jang.removeValue(forKey: "hadamard")
        }
        try fixture.withDirectory { directory in
            if reason == "sidecar-missing" {
                try FileManager.default.removeItem(
                    at: directory.appendingPathComponent("hadamard.json"))
            }
            #expect(throws: JangLoaderError.self) {
                try JangLoader.loadHadamardRuntimeContract(at: directory)
            }
        }
    }

    @Test(
        "malformed explicit markers never fall through to ordinary affine",
        arguments: ["string", "number", "null", "runtime-not-object"])
    func malformedMarkers(reason: String) throws {
        for key in [
            "requires_hadamard_activation_transform", "requires_jang_ternary_packed_expansion",
        ] {
            var fixture = JangHadamardFixture()
            fixture.config.removeValue(forKey: "hadamard")
            let value: Any
            switch reason {
            case "string": value = "true"
            case "number": value = 1
            default: value = NSNull()
            }
            fixture.jang = ["format": "jang", "runtime": [key: value]]
            if reason == "runtime-not-object" { fixture.jang["runtime"] = "invalid" }
            try fixture.withDirectory { directory in
                #expect(throws: JangLoaderError.self) {
                    if key == "requires_hadamard_activation_transform" {
                        _ = try JangLoader.loadHadamardRuntimeContract(at: directory)
                    } else {
                        _ = try JangLoader.loadTernaryPackedRuntimeContract(at: directory)
                    }
                }
            }
        }
    }

    @Test(
        "packed marker and manifest must agree",
        arguments: [
            "missing-marker", "missing-expansion", "empty-storage", "count",
            "wrong-runtime-bits", "wrong-group", "lossy", "changes-scales", "stored-bias",
            "entry-bits", "entry-storage", "affine1-conflict",
        ])
    func malformedPacked(reason: String) throws {
        var fixture = JangHadamardFixture(packed: true)
        var runtime = fixture.jang["runtime"] as! [String: Any]
        var quantization = fixture.jang["quantization"] as! [String: Any]
        var expansion = quantization["ternary_packed_runtime_expansion"] as! [String: Any]
        var manifest = quantization["tensor_quantization_manifest"] as! [String: Any]
        var entry = manifest[JangHadamardFixture.forward] as! [String: Any]
        switch reason {
        case "missing-marker": runtime.removeValue(forKey: "requires_jang_ternary_packed_expansion")
        case "wrong-runtime-bits": expansion["runtime_bits"] = 4
        case "wrong-group": expansion["group_size"] = 64
        case "lossy": expansion["lossless"] = false
        case "changes-scales": expansion["scales_unchanged"] = false
        case "stored-bias": expansion["biases"] = "stored"
        case "entry-bits": entry["bits"] = 3
        case "entry-storage": entry["storage"] = "unknown-packed"
        case "affine1-conflict": runtime["requires_jang_affine1_expansion"] = true
        default: break
        }
        manifest[JangHadamardFixture.forward] = entry
        if reason == "empty-storage" {
            for path in Array(manifest.keys) {
                var value = manifest[path] as! [String: Any]
                value.removeValue(forKey: "storage")
                manifest[path] = value
            }
        }
        quantization["ternary_packed_runtime_expansion"] =
            reason == "missing-expansion" ? NSNull() : expansion
        quantization["tensor_quantization_manifest"] = manifest
        if reason == "count" { quantization["tensor_quantization_manifest_count"] = 99 }
        fixture.jang["runtime"] = runtime
        fixture.jang["quantization"] = quantization
        try fixture.withDirectory { directory in
            #expect(throws: JangLoaderError.self) {
                try JangLoader.loadTernaryPackedRuntimeContract(at: directory)
            }
        }
    }
}
