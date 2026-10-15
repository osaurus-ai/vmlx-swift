import Foundation
import XCTest
import vMLXFluxKit
import vMLXFluxModels

final class ImageBatchTests: XCTestCase {
    func testRealQwenBatchMatchesSinglesAndEncodesOnce() async throws {
        guard let path = ProcessInfo.processInfo.environment["VMLX_FLUX_BATCH_MODEL"] else {
            throw XCTSkip("Set VMLX_FLUX_BATCH_MODEL for real-bundle parity and timing")
        }
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("flux-batch-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: out) }
        let model = try await QwenImage21(modelPath: URL(fileURLWithPath: path), quantize: nil)
        func collect(_ request: ImageGenRequest) async throws -> [URL] {
            var urls: [URL] = []
            var expectedIndex = 0
            for try await event in model.generate(request) {
                switch event {
                case .completed(let url, _, _): urls.append(url)
                case .imageCompleted(let url, let seed, let index, _):
                    XCTAssertEqual(index, expectedIndex)
                    XCTAssertEqual(seed, 7 + UInt64(index))
                    expectedIndex += 1
                    urls.append(url)
                case .failed(let message, _): XCTFail(message)
                case .cancelled: XCTFail("Unexpected cancellation")
                default: break
                }
            }
            return urls
        }
        var request = ImageGenRequest(prompt: "A red cube on a white table", width: 512, height: 512,
            steps: 8, guidance: 1, seed: 7, outputDir: out)
        _ = try await collect(request) // Compile/warm the exact shape before timing.
        let singleStart = Date()
        let beforeSingles = model.promptEncodingCount
        var singles: [Data] = []
        for seed in UInt64(7)...10 {
            request.seed = seed
            let urls = try await collect(request)
            XCTAssertEqual(urls.count, 1)
            singles.append(try Data(contentsOf: XCTUnwrap(urls.first)))
        }
        let singleSeconds = Date().timeIntervalSince(singleStart)
        XCTAssertEqual(model.promptEncodingCount - beforeSingles, 4)
        request.seed = 7
        request.numImages = 4
        let beforeBatch = model.promptEncodingCount
        let batchStart = Date()
        let batch = try await collect(request)
        let batchSeconds = Date().timeIntervalSince(batchStart)
        XCTAssertEqual(model.promptEncodingCount - beforeBatch, 1)
        XCTAssertEqual(batch.count, 4)
        for (index, url) in batch.enumerated() {
            XCTAssertEqual(try Data(contentsOf: url), singles[index], "Seed \(7 + index) byte parity")
        }
        print("FLUX_BATCH_PARITY singles_s=\(singleSeconds) batch_s=\(batchSeconds) images=4 encoder_calls=4:1")
        XCTAssertLessThan(batchSeconds, singleSeconds)
    }

    func testSequentialSeedsPreserveExistingWrappingContract() throws {
        let request = ImageGenRequest(prompt: "cube", seed: UInt64.max - 1,
            numImages: 4, outputDir: FileManager.default.temporaryDirectory)
        let plan = try ImageBatchPlan(request, maximumImages: 8, maximumParallelImages: 1)
        XCTAssertEqual(plan.seeds, [UInt64.max - 1, UInt64.max, 0, 1])
        XCTAssertEqual(plan.batchSize, 1)
    }

    func testExplicitSeedsPreserveOrderDuplicatesAndFullPrecision() throws {
        let seeds: [UInt64] = [7, UInt64.max, 7, 0]
        let request = ImageGenRequest(prompt: "cube", seed: 3, numImages: 4,
            seeds: seeds, batchSize: 4, outputDir: FileManager.default.temporaryDirectory)
        let plan = try ImageBatchPlan(request, maximumImages: 8, maximumParallelImages: 1)
        XCTAssertEqual(plan.seeds, seeds)
        XCTAssertEqual(plan.batchSize, 1)
    }

    func testMalformedBatchRejectedBeforeModelPreparation() throws {
        var request = ImageGenRequest(prompt: "cube", numImages: 4,
            outputDir: FileManager.default.temporaryDirectory)
        for count in [-1, 0, 9, Int.max] {
            request.numImages = count
            XCTAssertThrowsError(try ImageBatchPlan.validate(request, maximumImages: 8))
        }
        request.numImages = 4
        request.seeds = [1, 2]
        XCTAssertThrowsError(try ImageBatchPlan.validate(request, maximumImages: 8))
        request.seeds = [1, 2, 3, 4]
        request.batchSize = 0
        XCTAssertThrowsError(try ImageBatchPlan.validate(request, maximumImages: 8))
        request.batchSize = 1
        try ImageBatchPlan.validate(request, maximumImages: 8)
    }

    func testBatchCapabilitiesDoNotLoadWeights() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("{}".utf8).write(to: url.appendingPathComponent("model_index.json"))
        let limits = try QwenImage21.requestLimits(modelPath: url)
        XCTAssertEqual(limits.maximumImages, 8)
        XCTAssertEqual(limits.maximumParallelImages, 1)
        try limits.validate(ImageGenRequest(prompt: "cube", numImages: 4, outputDir: url))
    }
}
