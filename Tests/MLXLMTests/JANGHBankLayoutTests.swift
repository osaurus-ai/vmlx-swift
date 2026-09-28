import Cmlx
import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class JANGHBankLayoutTests: XCTestCase {
    private let prefix = "model.layers.0.mlp.switch_mlp"

    private func contract() throws -> JANGHFormatContract {
        let book: [String: Any] = [
            "alpha": 1.0, "beta": 0.0,
            "levels": (0 ..< 8).map { Double($0) - 3.5 },
        ]
        let root: [String: Any] = [
            "jangtq": [
                "version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                "codebook_family": "odd-cubic", "rotation": "none", "codebooks": ["3": book],
            ],
            "quantization": Dictionary(
                uniqueKeysWithValues: ["gate_proj", "up_proj", "down_proj"].map {
                    (
                        prefix + "." + $0,
                        ["mode": "jangtq2", "bits": 3, "rotation": "none"] as [String: Any]
                    )
                }),
        ]
        return try JANGHFormatContract(configuration: JSONSerialization.data(withJSONObject: root))
    }

    private func available(_ array: MLXArray) -> Bool {
        var value = false
        XCTAssertEqual(_mlx_array_is_available(&value, array.ctx), 0)
        return value
    }

    private func rowContiguous(_ array: MLXArray) -> Bool {
        var value = false
        XCTAssertEqual(_mlx_array_is_row_contiguous(&value, array.ctx), 0)
        return value
    }

    /// Four slots permit independently checking gate/up packed and scale banks.
    private func calls(_ banks: [MLXArray]) throws -> [() throws -> MLXArray] {
        let c = try contract()
        let single = try JANGHProjectionKernel(contract: c, module: prefix + ".gate_proj")
        let fused = try JANGHFusedGateUpKernel(
            contract: c, gateModule: prefix + ".gate_proj",
            upModule: prefix + ".up_proj", outputRotation: .none)
        let down = try JANGHWeightedDownKernel(contract: c, module: prefix + ".down_proj")
        let x = MLXArray([Float](repeating: 1, count: 32), [1, 32])
        let hidden = MLXArray([Float](repeating: 1, count: 64), [2, 32])
        let indices = MLXArray([UInt32(1), 0], [1, 2])
        let scores = MLXArray([Float(0.5), 0.5], [1, 2])
        return [
            { try single.project(x, packed: banks[0], scales: banks[1], indices: indices) },
            {
                try fused.activatePreparedInput(
                    x, gatePacked: banks[0], gateScales: banks[1],
                    upPacked: banks[2], upScales: banks[3],
                    indices: indices, limit: nil)
            },
            {
                try down.projectPreparedHidden(
                    hidden, preparedBasis: .none, packed: banks[0],
                    scales: banks[1], indices: indices, scores: scores,
                    outputDType: .float32)
            },
        ]
    }

    private func readyBanks() -> [MLXArray] {
        let packed = MLXArray([UInt32](repeating: 0, count: 54), [2, 9, 3])
        let scales = MLXArray([Float16](repeating: 1, count: 18), [2, 9])
        return [packed, scales, packed, scales]
    }

    private func expectRejected(_ body: () throws -> MLXArray, containing reason: String) {
        XCTAssertThrowsError(try body()) { error in
            guard case JANGHFormatContract.ValidationError.invalid(let message) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertTrue(message.contains(reason), message)
        }
    }

    func testReadyDenseAndOffsetBanksRemainAccepted() throws {
        try MLXMetalTestLock.withLock {
            let dense = readyBanks()
            for bank in dense {
                XCTAssertTrue(available(bank))
                XCTAssertTrue(rowContiguous(bank))
            }
            let denseOutputs = try calls(dense).map { try $0().asArray(Float.self) }
            XCTAssertTrue(denseOutputs[0].allSatisfy { $0 == -112 })
            XCTAssertTrue(denseOutputs[1].allSatisfy(\.isFinite))
            XCTAssertTrue(denseOutputs[2].allSatisfy { $0 == -112 })

            let packedBase = MLXArray([UInt32](repeating: 0, count: 81), [3, 9, 3])
            let scaleBase = MLXArray([Float16](repeating: 1, count: 27), [3, 9])
            let packedView = packedBase[1 ..< 3, 0..., 0...]
            let scaleView = scaleBase[1 ..< 3, 0...]
            // Only this tiny test view is explicitly made ready; the primitive never evaluates banks.
            eval(packedView, scaleView)
            XCTAssertTrue(rowContiguous(packedView))
            XCTAssertTrue(rowContiguous(scaleView))
            let offsetOutputs = try calls([packedView, scaleView, packedView, scaleView]).map {
                try $0().asArray(Float.self)
            }
            XCTAssertEqual(offsetOutputs, denseOutputs)
        }
    }

    func testSameShapeNoncontiguousPackedAndScaleBanksRefuse() throws {
        try MLXMetalTestLock.withLock {
            let packed = MLXArray([UInt32](repeating: 0, count: 54), [2, 3, 9]).transposed(0, 2, 1)
            let scales = MLXArray([Float16](repeating: 1, count: 18), [9, 2]).transposed()
            eval(packed, scales)
            XCTAssertEqual(packed.shape, [2, 9, 3])
            XCTAssertEqual(scales.shape, [2, 9])
            XCTAssertTrue(available(packed))
            XCTAssertTrue(available(scales))
            XCTAssertFalse(rowContiguous(packed))
            XCTAssertFalse(rowContiguous(scales))
            for slot in 0 ..< 4 {
                var banks = readyBanks()
                banks[slot] = slot.isMultiple(of: 2) ? packed : scales
                let candidates = try calls(banks)
                for index in slot < 2 ? [0, 1, 2] : [1] {
                    expectRejected(candidates[index], containing: "not row contiguous")
                }
            }
        }
    }

    func testUnavailableBanksAreNotEvaluatedOnRejection() throws {
        try MLXMetalTestLock.withLock {
            for slot in 0 ..< 4 {
                var banks = readyBanks()
                let lazy =
                    slot.isMultiple(of: 2)
                    ? MLXArray.zeros([2, 9, 3], dtype: .uint32)
                    : MLXArray.ones([2, 9], dtype: .float16)
                XCTAssertFalse(available(lazy))
                banks[slot] = lazy
                let candidates = try calls(banks)
                for index in slot < 2 ? [0, 1, 2] : [1] {
                    expectRejected(candidates[index], containing: "unavailable")
                    XCTAssertFalse(available(lazy), "Metadata guard must not evaluate the bank")
                }
            }
        }
    }
    private func withMmapPolicy<T>(enabled: Bool, body: () throws -> T) rethrows -> T {
        let settings = [
            "MLX_SAFETENSORS_MMAP": enabled ? "1" : "0",
            "VMLINUX_MMAP_SAFETENSORS": "0",
            "MLX_SAFETENSORS_MMAP_START_COLD": "0",
            "VMLINUX_MMAP_SAFETENSORS_START_COLD": "0",
            "MLX_SAFETENSORS_MMAP_TENSOR_BUFFERS": "0",
            "VMLINUX_MMAP_SAFETENSORS_TENSOR_BUFFERS": "0",
            "VMLX_MMAP_SAFETENSORS_TENSOR_BUFFERS": "0",
        ]
        let previous = settings.keys.map { ($0, getenv($0).map { String(cString: $0) }) }
        for (key, value) in settings { setenv(key, value, 1) }
        defer {
            for (key, value) in previous {
                if let value { setenv(key, value, 1) } else { unsetenv(key) }
            }
        }
        return try body()
    }

    /// Writes format bytes directly; test banks must come from the loader, not
    /// from copied MLXArray payloads. Data offsets are dtype aligned; the
    /// header places the packed payload at a 4096-byte boundary.
    private func alignedFixture(at url: URL) throws -> (String, String) {
        let packedKey = prefix + ".gate_proj.tq2_packed"
        let scalesKey = prefix + ".gate_proj.tq2_scales"
        let header: [String: Any] = [
            packedKey: ["dtype": "U32", "shape": [2, 9, 3], "data_offsets": [0, 216]],
            scalesKey: ["dtype": "F16", "shape": [2, 9], "data_offsets": [216, 252]],
        ]
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        let headerBytes = 4096 - 8
        guard json.count <= headerBytes else { throw NSError(domain: "JANGHFixture", code: 1) }
        json.append(Data(repeating: 32, count: headerBytes - json.count))
        var size = UInt64(headerBytes).littleEndian
        var file = Data()
        withUnsafeBytes(of: &size) { file.append(contentsOf: $0) }
        file.append(json)
        file.append(Data(repeating: 0, count: 216))
        for _ in 0 ..< 18 {
            var one = Float16(1).bitPattern.littleEndian
            withUnsafeBytes(of: &one) { file.append(contentsOf: $0) }
        }
        try file.write(to: url)
        return (packedKey, scalesKey)
    }

    func testAlignedMappedSafetensorsBanksAreReadyWithoutEvaluation() throws {
        try MLXMetalTestLock.withLock {
            let file = FileManager.default.temporaryDirectory
                .appendingPathComponent("jangh-mapped-" + UUID().uuidString + ".safetensors")
            defer { try? FileManager.default.removeItem(at: file) }
            let (packedKey, scalesKey) = try alignedFixture(at: file)
            try withMmapPolicy(enabled: true) {
                let before = mlx_safetensors_mmap_tracked_buffer_bytes()
                let arrays = try MLX.loadArrays(url: file, stream: .cpu)
                let packed = try XCTUnwrap(arrays[packedKey])
                let scales = try XCTUnwrap(arrays[scalesKey])
                XCTAssertGreaterThan(
                    mlx_safetensors_mmap_tracked_buffer_bytes(), before,
                    "Reader fallback must not masquerade as mapped storage")
                // All checks before the numerical dispatch are metadata-only.
                // No eval/asData/asArray/contiguous call prepares either bank.
                XCTAssertTrue(available(packed))
                XCTAssertTrue(available(scales))
                XCTAssertTrue(rowContiguous(packed))
                XCTAssertTrue(rowContiguous(scales))
                try JANGHBankLayout.requireReadyRowContiguous(packed, role: "mapped packed")
                try JANGHBankLayout.requireReadyRowContiguous(scales, role: "mapped scales")
                let op = try JANGHProjectionKernel(
                    contract: contract(), module: prefix + ".gate_proj")
                let result = try op.project(
                    MLXArray([Float](repeating: 1, count: 32), [1, 32]), packed: packed,
                    scales: scales, indices: MLXArray([UInt32(1), 0]))
                XCTAssertEqual(result.shape, [2, 9])
                // Code0 -> level -3.5; 32 unit inputs and scale1 -> -112.
                XCTAssertTrue(result.asArray(Float.self).allSatisfy { $0 == -112 })
            }
        }
    }

    func testOrdinaryReaderBanksRemainLazyAndRefuseImplicitPreparation() throws {
        try MLXMetalTestLock.withLock {
            let file = FileManager.default.temporaryDirectory
                .appendingPathComponent("jangh-reader-" + UUID().uuidString + ".safetensors")
            defer { try? FileManager.default.removeItem(at: file) }
            let (packedKey, scalesKey) = try alignedFixture(at: file)
            try withMmapPolicy(enabled: false) {
                let arrays = try MLX.loadArrays(url: file, stream: .cpu)
                let packed = try XCTUnwrap(arrays[packedKey])
                let scales = try XCTUnwrap(arrays[scalesKey])
                XCTAssertFalse(available(packed))
                XCTAssertFalse(available(scales))
                let op = try JANGHProjectionKernel(
                    contract: contract(), module: prefix + ".gate_proj")
                expectRejected(
                    {
                        try op.project(
                            MLXArray([Float](repeating: 1, count: 32), [1, 32]),
                            packed: packed, scales: scales, indices: MLXArray([UInt32(0)]))
                    }, containing: "unavailable")
                XCTAssertFalse(available(packed))
                XCTAssertFalse(available(scales))
            }
        }
    }

}
