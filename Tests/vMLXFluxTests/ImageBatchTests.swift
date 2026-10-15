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
        // Alternate order so a clock/thermal drift cannot consistently favor one arm.
        // Keep the short acceptance prompt and compare every output, not only timing.
        let repeats = max(1, Int(ProcessInfo.processInfo.environment["VMLX_FLUX_BATCH_REPEATS"] ?? "3") ?? 3)
        var singleTimes: [Double] = [], batchTimes: [Double] = []
        for trial in 0..<repeats {
            var outputs: [[Data]] = [[], []]
            for batched in (trial % 2 == 0 ? [false, true] : [true, false]) {
                request.numImages = batched ? 4 : 1
                request.seed = 7
                let before = model.promptEncodingCount
                let start = Date()
                print("FLUX_BATCH_ARM_START trial=\(trial) batch=\(batched) epoch=\(start.timeIntervalSince1970)")
                var urls: [URL] = []
                if batched {
                    urls = try await collect(request)
                } else {
                    for seed in UInt64(7)...10 {
                        request.seed = seed
                        urls += try await collect(request)
                    }
                }
                let elapsed = Date().timeIntervalSince(start)
                XCTAssertEqual(model.promptEncodingCount - before, batched ? 1 : 4)
                XCTAssertEqual(urls.count, 4)
                outputs[batched ? 1 : 0] = try urls.map { try Data(contentsOf: $0) }
                if batched { batchTimes.append(elapsed) } else { singleTimes.append(elapsed) }
                print("FLUX_BATCH_ARM_END trial=\(trial) batch=\(batched) seconds=\(elapsed) epoch=\(Date().timeIntervalSince1970)")
            }
            XCTAssertEqual(outputs[0], outputs[1], "All four seeds must remain byte-identical on trial \(trial)")
        }
        let singleMedian = singleTimes.sorted()[singleTimes.count / 2]
        let batchMedian = batchTimes.sorted()[batchTimes.count / 2]
        print("FLUX_BATCH_PARITY singles_s=\(singleTimes) batch_s=\(batchTimes) images=4 encoder_calls=4:1")
        XCTAssertLessThan(batchMedian, singleMedian)

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
