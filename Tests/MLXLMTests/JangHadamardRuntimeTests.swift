// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

private final class HadamardFixtureModel: Module, LanguageModel, JangHadamardRuntimeModel {
    @ModuleInfo var projection: Linear
    @ModuleInfo var embedding: Embedding
    @ParameterInfo var gain: MLXArray

    override init() {
        _projection.wrappedValue = Linear(512, 4, bias: false)
        _embedding.wrappedValue = Embedding(embeddingCount: 4, dimensions: 512)
        _gain.wrappedValue = MLXArray.ones([512], dtype: .float32)
    }

    func validateJangHadamardRuntime() throws {}
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    func callAsFunction(_ input: MLXArray, cache: [KVCache]?) -> MLXArray {
        projection(embedding(input) * gain)
    }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
}

@Suite("JANG Bonsai2 runtime primitives and load path", .serialized)
struct JangHadamardRuntimeTests {
    private static func trits(rows: Int, width: Int) -> [UInt8] {
        (0 ..< rows * width).map { UInt8(($0 * 17 + $0 / 13) % 3) }
    }

    /// Independent scalar encoder: it shares no MLX operations with expansion.
    private static func pack(_ trits: [UInt8]) -> [UInt8] {
        precondition(trits.count % 128 == 0)
        var result: [UInt8] = []
        for group in stride(from: 0, to: trits.count, by: 128) {
            for offset in stride(from: 0, to: 128, by: 5) {
                var byte = 0
                var power = 1
                for lane in 0 ..< min(5, 128 - offset) {
                    byte += Int(trits[group + offset + lane]) * power
                    power *= 3
                }
                result.append(UInt8(byte))
            }
        }
        return result
    }

    private static func nativeWords(_ trits: [UInt8]) -> [UInt32] {
        stride(from: 0, to: trits.count, by: 16).map { base in
            (0 ..< 16).reduce(UInt32(0)) { $0 | (UInt32(trits[base + $1]) << (2 * $1)) }
        }
    }

    private static func weights(packed: Bool) -> [String: MLXArray] {
        let values = trits(rows: 4, width: 512)
        let weight =
            packed
            ? MLXArray(pack(values), [4, 4 * 26])
            : MLXArray(nativeWords(values), [4, 512 / 16])
        // Values deliberately not all BF16-exact, to catch lossy dtype policy.
        let scales = MLXArray((0 ..< 16).map { Float($0 + 1) * 0.0137 }, [4, 4]).asType(.float16)
        var result: [String: MLXArray] = [
            "gain": MLXArray(Array(repeating: Float(1.0031), count: 512))
        ]
        for path in [JangHadamardFixture.forward, JangHadamardFixture.inverse] {
            result["\(path).weight"] = weight
            result["\(path).scales"] = scales
            result["\(path).signs"] = MLXArray(JangHadamardFixture.signs)
            if !packed { result["\(path).biases"] = -scales }
        }
        return result
    }

    @Test(
        "chunked expansion is bit-identical for all trit positions",
        arguments: [128, 512, 1024, 5120, 17408])
    func packedRoundTrip(width: Int) throws {
        try MLXMetalTestLock.withLock {
            let rows = 3
            let values = Self.trits(rows: rows, width: width)
            let input = MLXArray(Self.pack(values), [rows, width / 128 * 26])
            let scales = MLXArray.ones([rows, width / 128], dtype: .float16) * Float(0.3125)
            let chunked = try expandJangTernaryPacked(
                input, scales: scales, maximumChunkCodes: width)
            let whole = try expandJangTernaryPacked(
                input, scales: scales, maximumChunkCodes: rows * width)
            #expect(chunked.weight.dtype == .uint32)
            #expect(chunked.weight.shape == [rows, width / 16])
            #expect(chunked.weight.asArray(UInt32.self) == Self.nativeWords(values))
            #expect(chunked.weight.asArray(UInt32.self) == whole.weight.asArray(UInt32.self))
            #expect(chunked.scales === scales)
            #expect(chunked.biases.dtype == .float16)
            #expect(MLX.all(chunked.biases .== -scales).item(Bool.self))
        }
    }

    @Test(
        "packed head and tail byte extrema expand canonically",
        arguments: [UInt8(0), UInt8(1), UInt8(2)])
    func packedExtrema(trit: UInt8) throws {
        try MLXMetalTestLock.withLock {
            let values = Array(repeating: trit, count: 128)
            let result = try expandJangTernaryPacked(
                MLXArray(Self.pack(values), [1, 26]),
                scales: MLXArray.ones([1, 1], dtype: .float16))
            #expect(result.weight.asArray(UInt32.self) == Self.nativeWords(values))
        }
    }

    @Test(
        "malformed packed bytes, tensor layouts and bias companions are refused",
        arguments: [
            "head", "tail", "weight-dtype", "weight-rank", "empty", "columns",
            "scale-dtype", "scale-shape", "stored-bias", "missing-module", "chunk-size",
        ])
    func invalidPacked(reason: String) throws {
        try MLXMetalTestLock.withLock {
            var bytes = Array(repeating: UInt8(0), count: 26)
            if reason == "head" { bytes[24] = 243 }
            if reason == "tail" { bytes[25] = 27 }
            var packed = MLXArray(bytes, [1, 26])
            var scales = MLXArray.ones([1, 1], dtype: .float16)
            switch reason {
            case "weight-dtype": packed = packed.asType(.uint32)
            case "weight-rank": packed = packed.reshaped(26)
            case "empty": packed = MLXArray.zeros([0, 26], dtype: .uint8)
            case "columns": packed = MLXArray.zeros([1, 25], dtype: .uint8)
            case "scale-dtype": scales = scales.asType(.float32)
            case "scale-shape": scales = MLXArray.ones([1, 2], dtype: .float16)
            default: break
            }
            if reason == "stored-bias" || reason == "missing-module" {
                var weights = ["projection.weight": packed, "projection.scales": scales]
                if reason == "stored-bias" {
                    weights["projection.biases"] = -scales
                } else {
                    weights.removeValue(forKey: "projection.weight")
                }
                let contract = JangTernaryPackedRuntimeContract(modulePaths: ["projection"])
                #expect(throws: JangLoaderError.self) { try contract.expand(weights: &weights) }
            } else {
                #expect(throws: JangLoaderError.self) {
                    try expandJangTernaryPacked(
                        packed, scales: scales,
                        maximumChunkCodes: reason == "chunk-size" ? 0 : 1_048_576)
                }
            }
        }
    }

    @Test("all canonical byte values at every position match scalar decoding on GPU and CPU")
    func exhaustivePackedBytes() throws {
        try MLXMetalTestLock.withLock {
            var encoded: [UInt8] = []
            var expected: [UInt32] = []
            for position in 0 ..< 26 {
                for byte in 0 ... (position == 25 ? 26 : 242) {
                    var group = Array(repeating: UInt8(0), count: 26)
                    group[position] = UInt8(byte)
                    encoded.append(contentsOf: group)
                    var trits: [UInt8] = []
                    for (index, value) in group.enumerated() {
                        var remainder = Int(value)
                        for _ in 0 ..< (index == 25 ? 3 : 5) {
                            trits.append(UInt8(remainder % 3))
                            remainder /= 3
                        }
                    }
                    expected.append(contentsOf: Self.nativeWords(trits))
                }
            }
            let rows = encoded.count / 26
            for device in [Device.gpu, Device.cpu] {
                try Device.withDefaultDevice(device) {
                    let result = try expandJangTernaryPacked(
                        MLXArray(encoded, [rows, 26]),
                        scales: MLXArray.ones([rows, 1], dtype: .float16),
                        maximumChunkCodes: 128 * 111)
                    #expect(result.weight.asArray(UInt32.self) == expected)
                }
            }
        }
    }

    @Test("every noncanonical byte is rejected, including later row chunks")
    func allInvalidPackedBytes() throws {
        try MLXMetalTestLock.withLock {
            for position in 0 ..< 26 {
                for value in (position == 25 ? 27 : 243) ... 255 {
                    var bytes = Array(repeating: UInt8(0), count: 52)
                    bytes[26 + position] = UInt8(value)
                    #expect(throws: JangLoaderError.self) {
                        try expandJangTernaryPacked(
                            MLXArray(bytes, [2, 26]),
                            scales: MLXArray.ones([2, 1], dtype: .float16),
                            maximumChunkCodes: 128)
                    }
                }
            }
        }
    }

    @Test("strided packed input ignores invalid padding and preserves scale identity")
    func stridedPackedBytes() throws {
        try MLXMetalTestLock.withLock {
            let rows = 3
            let width = 512
            let values = Self.trits(rows: rows, width: width)
            let bytes = Self.pack(values).flatMap { [$0, UInt8(255)] }
            let backing = MLXArray(bytes, [rows, width / 128 * 52])
            let packed = backing[0..., .stride(by: 2)]
            let scales = MLXArray.ones([rows, width / 128], dtype: .float16)
            let result = try expandJangTernaryPacked(
                packed, scales: scales, maximumChunkCodes: width)
            #expect(result.weight.asArray(UInt32.self) == Self.nativeWords(values))
            #expect(result.scales === scales)
        }
    }

    /// Explicit Sylvester butterfly reference, normalized once per block.
    private static func referenceTransform(
        _ input: [Float], signs: [Float], block: Int, inverse: Bool
    ) -> [Float] {
        var output = input
        if !inverse { output = zip(output, signs).map(*) }
        for base in stride(from: 0, to: output.count, by: block) {
            var strideSize = 1
            while strideSize < block {
                for offset in stride(from: 0, to: block, by: strideSize * 2) {
                    for lane in 0 ..< strideSize {
                        let index = base + offset + lane
                        let a = output[index]
                        let b = output[index + strideSize]
                        output[index] = a + b
                        output[index + strideSize] = a - b
                    }
                }
                strideSize *= 2
            }
        }
        let scale = 1 / sqrt(Float(block))
        output = output.map { $0 * scale }
        if inverse { output = zip(output, signs).map(*) }
        return output
    }

    @Test(
        "forward and inverse use the reference sign order and block normalization",
        arguments: [512, 1024, 2048, 4096])
    func activationContract(block: Int) throws {
        try MLXMetalTestLock.withLock {
            let values = (0 ..< block * 2).map { Float(($0 * 17) % 61 - 30) / 64 }
            let signs = (0 ..< block * 2).map { Float($0 % 7 < 3 ? -1 : 1) }
            for dtype: DType in [.float32, .float16, .bfloat16] {
                let input = MLXArray(values, [1, 2 * block]).asType(dtype)
                for inverse in [false, true] {
                    let actual = jangHadamardActivation(
                        input, signs: MLXArray(signs), blockSize: block, inverse: inverse)
                    let expected = MLXArray(
                        Self.referenceTransform(
                            values, signs: signs, block: block, inverse: inverse)
                    )
                    .reshaped(input.shape).asType(dtype)
                    #expect(actual.dtype == dtype)
                    #expect(actual.shape == input.shape)
                    let error = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(
                        Float.self)
                    #expect(error < (dtype == .float32 ? 1e-5 : dtype == .float16 ? 0.004 : 0.03))
                }
            }
            let input = MLXArray(values, [1, 2 * block])
            let rotated = jangHadamardActivation(input, signs: MLXArray(signs), blockSize: block)
            let restored = jangHadamardActivation(
                rotated, signs: MLXArray(signs), blockSize: block, inverse: true)
            #expect(abs(input - restored).max().item(Float.self) < 1e-5)
        }
    }

    @Test("wrappers preserve quantized arrays and cannot enter raw projection fusion")
    func wrappersReuseArrays() throws {
        try MLXMetalTestLock.withLock {
            let values = Self.weights(packed: false)
            let plain = QuantizedLinear(
                weight: values["projection.weight"]!, scales: values["projection.scales"]!,
                biases: values["projection.biases"], groupSize: 128, bits: 2)
            let rotated = HadamardQuantizedLinear(plain, blockSize: 512)
            rotated.update(
                parameters: ModuleParameters.unflattened([
                    ("signs", MLXArray(JangHadamardFixture.signs))
                ]))
            #expect(rotated.weight === plain.weight)
            #expect(rotated.scales === plain.scales)
            #expect(rotated.biases === plain.biases)
            #expect(jangAllowsRawQuantizedProjection(plain))
            #expect(!jangAllowsRawQuantizedProjection(rotated))
            let input = MLXArray((0 ..< 512).map { Float($0 % 19 - 9) * 0.125 }, [1, 512])
            let transformed = jangHadamardActivation(input, signs: rotated.signs, blockSize: 512)
            #expect(MLX.all(rotated(input) .== plain(transformed)).item(Bool.self))
            #expect(abs(rotated(input) - plain(input)).max().item(Float.self) > 0.01)

            let ordinaryEmbedding = QuantizedEmbedding(
                weight: values["embedding.weight"]!, scales: values["embedding.scales"]!,
                biases: values["embedding.biases"], groupSize: 128, bits: 2)
            let embedding = HadamardQuantizedEmbedding(ordinaryEmbedding, blockSize: 512)
            embedding.update(parameters: ModuleParameters.unflattened([("signs", rotated.signs)]))
            #expect(embedding.weight === ordinaryEmbedding.weight)
            #expect(embedding.scales === ordinaryEmbedding.scales)
            let ids = MLXArray([Int32(1), 3], [1, 2])
            let expected = jangHadamardActivation(
                ordinaryEmbedding(ids), signs: embedding.signs, blockSize: 512, inverse: true)
            #expect(MLX.all(embedding(ids) .== expected).item(Bool.self))
        }
    }

    @Test("loadWeights preserves the validated Hadamard dtypes and packed/affine logits")
    func realLoadPath() throws {
        try MLXMetalTestLock.withLock {
            var outputs: [[Float]] = []
            for packed in [false, true] {
                try JangHadamardFixture(packed: packed).withDirectory { directory in
                    let tensors = Self.weights(packed: packed)
                    try MLX.save(
                        arrays: tensors, url: directory.appendingPathComponent("model.safetensors"))
                    let model = HadamardFixtureModel()
                    try loadWeights(
                        modelDirectory: directory, model: model,
                        quantization: .init(groupSize: 128, bits: 2),
                        jangConfig: try JangLoader.loadConfig(at: directory))
                    let projection = try #require(model.projection as? HadamardQuantizedLinear)
                    let embedding = try #require(model.embedding as? HadamardQuantizedEmbedding)
                    #expect(projection.scales.dtype == .float16)
                    #expect(embedding.scales.dtype == .float16)
                    #expect(projection.signs.dtype == .float32)
                    #expect(model.gain.dtype == .float32)
                    #expect(model.gain.asArray(Float.self) == tensors["gain"]!.asArray(Float.self))
                    #expect(
                        projection.scales.asArray(Float16.self)
                            == tensors["projection.scales"]!.asArray(Float16.self))
                    let output = model(MLXArray([Int32(1), 2], [1, 2]), cache: nil)
                    let values = output.asType(.float32).asArray(Float.self)
                    let allFinite = values.allSatisfy(\.isFinite)
                    #expect(allFinite)
                    outputs.append(values)
                }
            }
            #expect(outputs[0] == outputs[1])
        }
    }

    @Test("ordinary affine load retains the existing bf16 materialization policy")
    func ordinaryLoadPathUnchanged() throws {
        try MLXMetalTestLock.withLock {
            var fixture = JangHadamardFixture()
            fixture.config.removeValue(forKey: "hadamard")
            fixture.jang = [
                "format": "jang", "format_version": "2.0", "weight_format": "affine",
                "quantization": ["bits": 2, "block_size": 128, "bit_widths_used": [2]],
            ]
            try fixture.withDirectory { directory in
                var tensors = Self.weights(packed: false)
                tensors = tensors.filter { !$0.key.hasSuffix(".signs") }
                try MLX.save(
                    arrays: tensors, url: directory.appendingPathComponent("model.safetensors"))
                let model = HadamardFixtureModel()
                try loadWeights(
                    modelDirectory: directory, model: model,
                    quantization: .init(groupSize: 128, bits: 2),
                    jangConfig: try JangLoader.loadConfig(at: directory))
                let projection = try #require(model.projection as? QuantizedLinear)
                #expect(!(projection is HadamardQuantizedLinear))
                #expect(projection.scales.dtype == .bfloat16)
                #expect(model.gain.dtype == .bfloat16)
            }
        }
    }

    @Test(
        "missing signs and architecture-incompatible weights fail the production load",
        arguments: ["missing", "zeros", "wrong-signs", "wrong-shape", "wrong-dtype"])
    func invalidLoadedSigns(reason: String) throws {
        try MLXMetalTestLock.withLock {
            try JangHadamardFixture().withDirectory { directory in
                var tensors = Self.weights(packed: false)
                switch reason {
                case "missing": tensors.removeValue(forKey: "projection.signs")
                case "zeros": tensors["projection.signs"] = MLXArray.zeros([512], dtype: .float32)
                case "wrong-signs":
                    tensors["projection.signs"] = MLXArray.ones([512], dtype: .float32)
                case "wrong-shape":
                    tensors["projection.signs"] = MLXArray.ones([1024], dtype: .float32)
                case "wrong-dtype":
                    tensors["projection.signs"] = MLXArray(JangHadamardFixture.signs).asType(
                        .float16)
                default: break
                }
                try MLX.save(
                    arrays: tensors, url: directory.appendingPathComponent("model.safetensors"))
                #expect(throws: JangLoaderError.self) {
                    try loadWeights(
                        modelDirectory: directory, model: HadamardFixtureModel(),
                        quantization: .init(groupSize: 128, bits: 2),
                        jangConfig: try JangLoader.loadConfig(at: directory))
                }
            }
        }
    }
}
