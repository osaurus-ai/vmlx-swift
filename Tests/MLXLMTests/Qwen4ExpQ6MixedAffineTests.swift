import Foundation
import CryptoKit
import Darwin
import MLX
import MLXLMCommon
import Testing

/// Generated fixtures only: no model is loaded and no bundle is modified.
/// The large shapes match the Flash-Next JANG_2L GDN header inventory.
@Suite("Qwen4Exp q6 mixed affine decode", .serialized)
struct Qwen4ExpQ6MixedAffineTests {
    private func fixture(k: Int, n: Int, rows: Int = 1, group: Int = 64, seed: UInt64)
        -> (MLXArray, MLXArray, MLXArray, MLXArray?)
    {
        let x = MLXRandom.normal([1, rows, k], key: MLXRandom.key(seed))
            .asType(.bfloat16)
        let w = (MLXRandom.normal([n, k], key: MLXRandom.key(seed + 1))
            / Float(k).squareRoot()).asType(.float16)
        let (q, s, b) = quantized(w, groupSize: group, bits: 6, mode: .affine)
        MLX.eval(x, q, s)
        if let b { MLX.eval(b) }
        return (x, q, s, b)
    }

    private func product(
        _ x: MLXArray, _ q: MLXArray, _ s: MLXArray, _ b: MLXArray?,
        group: Int = 64, reference: Bool = false
    ) -> MLXArray {
        quantizedMM(
            reference ? x.asType(.float32) : x, q,
            scales: reference ? s.asType(.float32) : s,
            biases: reference ? b?.asType(.float32) : b,
            transpose: true, groupSize: group, bits: 6, mode: .affine)
    }

    @Test("q6 decode matches the promoted-F32 reference at bundle projection shapes")
    func mixedDenseQ6Parity() {
        MLXMetalTestLock.withLock {
            for (k, n, seed) in [
                (512, 8, UInt64(0)), (2560, 48, 17),
                (2560, 10240, 829), (2560, 6144, 830),
                (2560, 16480, 831), (6144, 2560, 832),
            ] {
                let (x, q, s, b) = fixture(k: k, n: n, seed: seed)
                let actual = product(x, q, s, b)
                let expected = product(x, q, s, b, reference: true).asType(.bfloat16)
                MLX.eval(actual, expected)
                let rounded = actual.asType(.bfloat16)
                let error = abs(rounded.asType(.float32) - expected.asType(.float32))
                    .max().item(Float.self)
                let unequal = (rounded .!= expected).asType(.int32).sum().item(Int.self)
                print("[q6-mixed] K=\(k) N=\(n) seed=\(seed) input=\(x.dtype)"
                    + " metadata=\(s.dtype) output=\(actual.dtype) max_abs=\(error)"
                    + " unequal_bf16=\(unequal) generated_weights=1")
                #expect(actual.dtype == .bfloat16)
                #expect(actual.shape == [1, 1, n])
                #expect(isFinite(actual).all().item(Bool.self))
                // This is a storage/dispatch optimization, not permission to
                // add BF16 intermediate rounding to the existing F32 reduction.
                #expect(unequal == 0)
            }
        }
    }

    @Test("q6 unsupported shapes and metadata retain the existing promotion")
    func mixedDenseQ6Fallbacks() {
        MLXMetalTestLock.withLock {
            for (k, n, rows, group) in [
                (512, 8, 2, 64), (512, 8, 4, 64),
                (512, 8, 1, 32), (512, 8, 1, 128),
                (768, 8, 1, 64), (512, 7, 1, 64),
            ] {
                let (x, q, s, b) = fixture(k: k, n: n, rows: rows, group: group, seed: 91)
                let actual = product(x, q, s, b, group: group)
                let expected = product(x, q, s, b, group: group, reference: true)
                MLX.eval(actual, expected)
                #expect(actual.dtype == .float32)
                #expect((actual .== expected).all().item(Bool.self))
            }
            let (x, q, s, b) = fixture(k: 512, n: 8, seed: 92)
            let f32Metadata = product(x, q, s.asType(.float32), b?.asType(.float32))
            let f16Input = product(x.asType(.float16), q, s, b)
            MLX.eval(f32Metadata, f16Input)
            #expect(f32Metadata.dtype == .float32)
            #expect(f16Input.dtype == .float16)
            let cpu = quantizedMM(
                x, q, scales: s, biases: b, transpose: true,
                groupSize: 64, bits: 6, mode: .affine, stream: .cpu)
            let cpuReference = quantizedMM(
                x.asType(.float32), q, scales: s.asType(.float32),
                biases: b?.asType(.float32), transpose: true,
                groupSize: 64, bits: 6, mode: .affine, stream: .cpu)
            MLX.eval(cpu, cpuReference)
            #expect(cpu.dtype == .float32)
            #expect((cpu .== cpuReference).all().item(Bool.self))
        }
    }

    @Test("q6 strided input, compiled projection and verifier rows preserve results")
    func mixedDenseQ6LayoutsAndCompilation() {
        MLXMetalTestLock.withLock {
            let (_, q, s, b) = fixture(k: 2560, n: 48, seed: 123)
            let backing = MLXRandom.normal([4, 5120], key: MLXRandom.key(456))
                .asType(.bfloat16)
            let rows = backing[0..., .stride(by: 2)]
            MLX.eval(rows)
            let batched = product(rows, q, s, b).asType(.bfloat16)
            let rowwise = concatenated((0..<4).map {
                product(rows[$0..<($0 + 1)], q, s, b)
            })
            let compiled = MLX.compile { (x: MLXArray) in
                quantizedMM(x, q, scales: s, biases: b, transpose: true,
                    groupSize: 64, bits: 6, mode: .affine)
            }
            let oneRow = rows[0..<1]
            let compiledRow = compiled(oneRow)
            let reference = product(oneRow, q, s, b, reference: true).asType(.bfloat16)
            let modelRoute = Qwen4ExpBF16Affine.dense(
                oneRow, q, scales: s, biases: b,
                groupSize: 64, bits: 6, mode: .affine)
            MLX.eval(batched, rowwise, compiledRow, reference, modelRoute)
            #expect((batched .== rowwise).all().item(Bool.self))
            #expect(compiledRow.dtype == .bfloat16)
            #expect((compiledRow .== reference).all().item(Bool.self))
            #expect((modelRoute .== reference).all().item(Bool.self))
        }
    }

    @Test("opt-in actual q6 projection payload parity", .enabled(if:
        ProcessInfo.processInfo.environment["VMLX_Q6_BUNDLE_INVENTORY"] != nil))
    func actualPackedQ6Payloads() throws {
        try MLXMetalTestLock.withLock {
            let inventoryPath = try #require(
                ProcessInfo.processInfo.environment["VMLX_Q6_BUNDLE_INVENTORY"])
            let inventory = try #require(JSONSerialization.jsonObject(
                with: Data(contentsOf: URL(fileURLWithPath: inventoryPath))) as? [String: Any])
            let directory = URL(fileURLWithPath: try #require(inventory["bundle"] as? String))
            for (filename, key) in [
                ("config.json", "config_sha256"),
                ("model.safetensors.index.json", "index_sha256"),
            ] {
                let bytes = try Data(contentsOf: directory.appendingPathComponent(filename))
                let hash = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
                try #require(hash == inventory[key] as? String)
            }
            let modules = try #require(inventory["modules"] as? [[String: Any]])
            var count = 0
            var payloadBytes = 0
            func tensor(_ entry: [String: Any], name: String) throws -> MLXArray {
                let file = try #require(entry["file"] as? String)
                try #require(!file.contains("/") && file.hasSuffix(".safetensors"))
                let handle = try FileHandle(forReadingFrom: directory.appendingPathComponent(file))
                defer { try? handle.close() }
                try #require(fcntl(handle.fileDescriptor, F_NOCACHE, 1) == 0)
                func readExactly(_ length: Int) throws -> Data {
                    var data = Data()
                    while data.count < length {
                        let part = try #require(try handle.read(upToCount: length - data.count))
                        try #require(!part.isEmpty)
                        data.append(part)
                    }
                    return data
                }
                let prefix = try readExactly(8)
                let headerSize = prefix.withUnsafeBytes {
                    UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
                }
                try #require(headerSize > 0 && headerSize <= 2 * 1024 * 1024)
                let header = try #require(JSONSerialization.jsonObject(
                    with: readExactly(Int(headerSize))) as? [String: Any])
                let current = try #require(header[name] as? [String: Any])
                let offsets = try #require(current["data_offsets"] as? [Int])
                let shape = try #require(current["shape"] as? [Int])
                let dtypeName = try #require(current["dtype"] as? String)
                try #require(offsets == entry["data_offsets"] as? [Int])
                try #require(shape == entry["shape"] as? [Int])
                try #require(dtypeName == entry["dtype"] as? String)
                try #require(offsets.count == 2 && offsets[0] >= 0 && offsets[1] > offsets[0])
                let length = offsets[1] - offsets[0]
                try #require(length <= 64 * 1024 * 1024)
                try #require(shape.count == 2 && shape.allSatisfy { $0 > 0 && $0 <= 65536 })
                let dtype: DType = dtypeName == "U32" ? .uint32 : .float16
                try #require(dtypeName == "U32" || dtypeName == "F16")
                try #require(shape.reduce(1, *) * (dtype == .uint32 ? 4 : 2) == length)
                try handle.seek(toOffset: 8 + headerSize + UInt64(offsets[0]))
                let data = try readExactly(length)
                payloadBytes += length
                return MLXArray(data, shape, dtype: dtype)
            }
            for module in modules {
                let name = try #require(module["module"] as? String)
                guard name.contains(".linear_attn."),
                    let quant = module["quantization"] as? [String: Any],
                    quant["bits"] as? Int == 6, quant["group_size"] as? Int == 64,
                    quant["mode"] as? String == "affine"
                else { continue }
                let q = try tensor(try #require(module["weight"] as? [String: Any]), name: name + ".weight")
                let s = try tensor(try #require(module["scales"] as? [String: Any]), name: name + ".scales")
                let b = try tensor(try #require(module["biases"] as? [String: Any]), name: name + ".biases")
                let k = s.dim(1) * 64
                let n = q.dim(0)
                try #require(q.dtype == .uint32 && s.dtype == .float16 && b.dtype == .float16)
                try #require(s.shape == b.shape && s.dim(0) == n)
                try #require(q.dim(1) * 32 == k * 6 && k % 512 == 0 && n % 8 == 0)
                for seed: UInt64 in [0, 829, 65537] {
                    let x = MLXRandom.normal([1, 1, k], key: MLXRandom.key(seed)).asType(.bfloat16)
                    let actual = product(x, q, s, b)
                    let expected = product(x, q, s, b, reference: true).asType(.bfloat16)
                    MLX.eval(actual, expected)
                    #expect(actual.dtype == .bfloat16)
                    #expect(isFinite(actual).all().item(Bool.self))
                    #expect((actual .== expected).all().item(Bool.self), "\(name) seed=\(seed)")
                }
                count += 1
                print("[q6-actual] module=\(name) K=\(k) N=\(n) seeds=0,829,65537 compared=1")
            }
            #expect(count > 0)
            print("[q6-actual] projections=\(count) payload_bytes=\(payloadBytes)"
                + " no_full_model=1 no_weight_writes=1 no_cache_writes=1")
        }
    }

    @Test("diagnostic disable restores the original F32 projection", .enabled(if:
        ProcessInfo.processInfo.environment["VMLX_DISABLE_MIXED_Q6"] == "1"))
    func mixedQ6DisabledUsesPromotion() {
        MLXMetalTestLock.withLock {
            let (x, q, s, b) = fixture(k: 2560, n: 48, seed: 829)
            let actual = product(x, q, s, b)
            let expected = product(x, q, s, b, reference: true)
            MLX.eval(actual, expected)
            #expect(actual.dtype == .float32)
            #expect((actual .== expected).all().item(Bool.self))
            print("[q6-disabled] actual_output=\(actual.dtype) reference_exact=1")
        }
    }

    @Test("opt-in synchronized q6 GDN projection timing")
    func mixedDenseQ6Timing() {
        guard ProcessInfo.processInfo.environment["VMLX_Q6_MIXED_BENCH"] == "1" else {
            return
        }
        MLXMetalTestLock.withLock {
            for (k, n) in [(2560, 16480), (6144, 2560)] {
                let (x, q, s, b) = fixture(k: k, n: n, seed: 829)
                func run(_ reference: Bool) {
                    MLX.eval(product(x, q, s, b, reference: reference).asType(.bfloat16))
                }
                for _ in 0..<5 { run(false); run(true) }
                var mixed: [Double] = []
                var reference: [Double] = []
                for round in 0..<9 {
                    for isReference in (round % 2 == 0 ? [false, true] : [true, false]) {
                        let start = DispatchTime.now().uptimeNanoseconds
                        for _ in 0..<5 { run(isReference) }
                        let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 5e6
                        if isReference { reference.append(ms) } else { mixed.append(ms) }
                    }
                }
                print("[q6-mixed-bench] K=\(k) N=\(n) mixed_ms=\(mixed)"
                    + " promoted_ms=\(reference) ratio=\(reference.sorted()[4] / mixed.sorted()[4])"
                    + " generated_weights=1 cache_warm=1 synchronized_calls=1"
                    + " host_configuration=debug_not_model_tps")
            }
        }
    }
}
