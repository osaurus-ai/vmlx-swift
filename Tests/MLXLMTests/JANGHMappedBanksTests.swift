import Cmlx
import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class JANGHMappedBanksTests: XCTestCase {
    private let parent = "model.layers.0.mlp.switch_mlp"
    private let shard = "model.safetensors"
    private let index = "model.safetensors.index.json"

    private func fixture(_ directory: URL) throws -> (JANGHFormatContract, [String: JANGHTensorIndexPlan.Dimensions]) {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var books: [String: Any] = [:]
        var quant: [String: Any] = [:]
        var headers: [String: Any] = [:]
        var map: [String: String] = [:]
        var dimensions: [String: JANGHTensorIndexPlan.Dimensions] = [:]
        var payload = Data()
        for (role, bits) in [("gate_proj", 2), ("up_proj", 3), ("down_proj", 4)] {
            let module = parent + "." + role
            books[String(bits)] = ["alpha": 0.25, "beta": 0,
                                   "levels": (0..<(1 << bits)).map { (Double($0) - Double((1 << bits) - 1) / 2) * 0.25 }]
            quant[module] = ["mode": "jangtq2", "bits": bits, "rotation": "none"]
            dimensions[module] = .init(experts: 2, input: 32, output: 32)
            for (suffix, dtype, shape, data) in [
                ("tq2_packed", "U32", [2, 32, bits], Data(repeating: 0, count: 2 * 32 * bits * 4)),
                ("tq2_scales", "F16", [2, 32], Data((0..<64).flatMap { _ in [UInt8(0), UInt8(0x3c)] })),
            ] {
                let name = module + "." + suffix
                headers[name] = ["dtype": dtype, "shape": shape, "data_offsets": [payload.count, payload.count + data.count]]
                map[name] = shard
                payload.append(data)
            }
        }
        var header = try JSONSerialization.data(withJSONObject: headers, options: .sortedKeys)
        XCTAssertLessThan(header.count, 4088)
        header.append(Data(repeating: 0x20, count: 4088 - header.count))
        var length = UInt64(header.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(header)
        file.append(payload)
        try file.write(to: directory.appendingPathComponent(shard), options: .atomic)
        try JSONSerialization.data(withJSONObject: ["weight_map": map]).write(
            to: directory.appendingPathComponent(index), options: .atomic)
        let config: [String: Any] = ["jangtq": ["version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                                                       "codebook_family": "odd-cubic", "rotation": "none", "codebooks": books],
                                     "quantization": quant]
        return (try JANGHFormatContract(configuration: JSONSerialization.data(withJSONObject: config)), dimensions)
    }

    private func withFixture(_ body: (URL, JANGHFormatContract, [String: JANGHTensorIndexPlan.Dimensions]) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let (contract, dimensions) = try fixture(directory)
        try body(directory, contract, dimensions)
    }

    func testOwnedCompositionSurvivesBankOwnerReleaseAndUnlink() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { directory, contract, dimensions in
                let metadata = try JANGHHeaderAdapter.read(directory: directory, indexName: index)
                let before = mlx_safetensors_mmap_tracked_buffer_bytes()
                var owner: JANGHMappedBanks? = try JANGHMappedBanks(
                    directory: directory, metadata: metadata, contract: contract, dimensions: dimensions)
                // Production reads into owned allocations since 1747b2bdb.
                // A rising retired mmap counter would violate that contract.
                XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), before)
                let gate = try owner!.projection(parent + ".gate_proj")
                XCTAssertFalse(WeightBacking.sharesMappedShard(gate.packed))
                try JANGHBankLayout.requireReadyRowContiguous(gate.packed, role: "test packed")
                let block = try JANGHRoutedDecodeBlock(banks: owner!, parentModule: parent, activationLimit: 10)
                owner = nil
                try FileManager.default.removeItem(at: directory.appendingPathComponent(shard))
                let input = MLXArray(Array(repeating: Float(1), count: 32)).reshaped(1, 32)
                let output = try block(input, indices: MLXArray([UInt32(0), 1]).reshaped(1, 2),
                                       scores: MLXArray([Float(0.25), 0.75]).reshaped(1, 2), outputDType: .float32)
                // Independent scalar reference: code0 maps to centered linear levels;
                // gate=-12, up=-28→-10, down level=-1.875, two weights sum to1.
                let gateValue = -12.0
                let hidden = (gateValue / (1 + exp(-gateValue))) * -10
                let expected = Float(32 * -1.875 * hidden)
                for value in output.asArray(Float.self) { XCTAssertEqual(value, expected, accuracy: 1e-5) }
            }
        }
    }

    func testReplacementAndWrongGeometryFailBeforeAnyMapping() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { directory, contract, dimensions in
                let metadata = try JANGHHeaderAdapter.read(directory: directory, indexName: index)
                let before = mlx_safetensors_mmap_tracked_buffer_bytes()
                let bytes = try Data(contentsOf: directory.appendingPathComponent(shard))
                try bytes.write(to: directory.appendingPathComponent(shard), options: .atomic)
                XCTAssertThrowsError(try JANGHMappedBanks(directory: directory, metadata: metadata,
                                                          contract: contract, dimensions: dimensions))
                XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), before)
                let fresh = try JANGHHeaderAdapter.read(directory: directory, indexName: index)
                var wrong = dimensions
                wrong[parent + ".gate_proj"] = .init(experts: 3, input: 32, output: 32)
                XCTAssertThrowsError(try JANGHMappedBanks(directory: directory, metadata: fresh,
                                                          contract: contract, dimensions: wrong))
                XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), before)
            }
        }
    }

    func testOwnerRejectsChangedIndexAndBlockRejectsUnknownModule() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { directory, contract, dimensions in
                let metadata = try JANGHHeaderAdapter.read(directory: directory, indexName: index)
                let bytes = try Data(contentsOf: directory.appendingPathComponent(index))
                try bytes.write(to: directory.appendingPathComponent(index), options: .atomic)
                XCTAssertThrowsError(try JANGHMappedBanks(directory: directory, metadata: metadata,
                                                          contract: contract, dimensions: dimensions))
                let fresh = try JANGHHeaderAdapter.read(directory: directory, indexName: index)
                let owner = try JANGHMappedBanks(directory: directory, metadata: fresh,
                                               contract: contract, dimensions: dimensions)
                XCTAssertThrowsError(try JANGHRoutedDecodeBlock(banks: owner, parentModule: "missing", activationLimit: nil))
                XCTAssertThrowsError(try JANGHRoutedDecodeBlock(banks: owner, parentModule: parent, activationLimit: .infinity))
            }
        }
    }
    func testPathReplacementAfterSourceLeaseFailsClosedBeforeMapping() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { directory, contract, dimensions in
                let metadata = try JANGHHeaderAdapter.read(directory: directory, indexName: index)
                let lease = try JANGHMappedBanks.SourceLease(directory: directory, metadata: metadata,
                                                           contract: contract, dimensions: dimensions)
                let before = mlx_safetensors_mmap_tracked_buffer_bytes()
                let original = try Data(contentsOf: directory.appendingPathComponent(shard))
                // Atomic replacement unlinks the old inode; retained descriptors
                // still address that inode, whose ctime changes. Never map the replacement.
                try Data(repeating: 0xFF, count: original.count).write(
                    to: directory.appendingPathComponent(shard), options: .atomic)
                XCTAssertThrowsError(try JANGHMappedBanks(source: lease))
                XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), before)
            }
        }
    }

    func testDroppingBankOwnerReleasesOwnedActiveAllocations() throws {
        try MLXMetalTestLock.withLock {
            try withFixture { directory, contract, dimensions in
                let metadata = try JANGHHeaderAdapter.read(directory: directory, indexName: index)
                let before = mlx_safetensors_mmap_tracked_buffer_bytes()
                let activeBefore = MLX.Memory.snapshot().activeMemory
                var owner: JANGHMappedBanks? = try JANGHMappedBanks(
                    directory: directory, metadata: metadata, contract: contract, dimensions: dimensions)
                weak var releasedOwner = owner
                XCTAssertNotNil(owner)
                XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), before)
                XCTAssertGreaterThan(MLX.Memory.snapshot().activeMemory, activeBefore)
                owner = nil
                XCTAssertNil(releasedOwner)
                // Reusable allocator cache bytes are distinct from live weights.
                XCTAssertEqual(MLX.Memory.snapshot().activeMemory, activeBefore)
                XCTAssertEqual(mlx_safetensors_mmap_tracked_buffer_bytes(), before)
            }
        }
    }

}
