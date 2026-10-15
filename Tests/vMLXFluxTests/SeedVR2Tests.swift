import Foundation
import MLX
import XCTest
import vMLXFlux
import vMLXFluxModels

@testable import vMLXSeedVR2

final class SeedVR2Tests: XCTestCase {
    func testBundleRejectsUnsupportedVariantBeforeWeightLoad() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"model_type":"seedvr2","variant":"seedvr2-7b"}"#.utf8)
            .write(to: dir.appendingPathComponent("config.json"))
        XCTAssertThrowsError(try SeedVR2Bundle(directory: dir))
    }

    func testBundleMetadataDetectsRenamedExportAndMissingEmbedding() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(
            #"{"model_type":"seedvr2","variant":"seedvr2-3b","quantization":{"bits":8,"group_size":64}}"#
                .utf8
        )
        .write(to: dir.appendingPathComponent("config.json"))
        for name in ["transformer.safetensors", "vae.safetensors", "pos_emb.safetensors"] {
            try Data().write(to: dir.appendingPathComponent(name))
        }
        let facts = try MLXStudioModelStore.inspect(directory: dir)
        XCTAssertEqual(facts.canonicalName, "seedvr2")
        XCTAssertEqual(facts.kind, .imageUpscale)
        XCTAssertEqual(facts.quantizationBits, 8)
        XCTAssertTrue(facts.blockedReasons.isEmpty)
        try FileManager.default.removeItem(at: dir.appendingPathComponent("pos_emb.safetensors"))
        XCTAssertFalse(try MLXStudioModelStore.inspect(directory: dir).blockedReasons.isEmpty)
    }

    func testUpscaleValidationRejectsBeforeAllocation() throws {
        XCTAssertThrowsError(
            try SeedVR2BundleMetadata.validateUpscale(
                sourceWidth: 2_000_000, sourceHeight: 1, scale: 2, format: .png))
        for scale in [0, 1, 3, Int.max] {
            XCTAssertThrowsError(
                try SeedVR2BundleMetadata.validateUpscale(
                    sourceWidth: 512, sourceHeight: 512, scale: scale, format: .png))
        }
        XCTAssertThrowsError(
            try SeedVR2BundleMetadata.validateUpscale(
                sourceWidth: Int.max, sourceHeight: 512, scale: 4, format: .png))
        XCTAssertThrowsError(
            try SeedVR2BundleMetadata.validateUpscale(
                sourceWidth: 2048, sourceHeight: 2048, scale: 4, format: .png))
        XCTAssertThrowsError(
            try SeedVR2BundleMetadata.validateUpscale(
                sourceWidth: 512, sourceHeight: 512, scale: 2, format: .jpeg))
        try SeedVR2BundleMetadata.validateUpscale(
            sourceWidth: 1024, sourceHeight: 1024, scale: 4, format: .png)
    }

    /// Opt-in real exported weights + independently saved mflux tensors, no downloads in tests.
    func testReferenceComponents() throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundle = env["SEEDVR2_TEST_BUNDLE"], let fixture = env["SEEDVR2_TEST_FIXTURE"]
        else {
            throw XCTSkip("Set SEEDVR2_TEST_BUNDLE and SEEDVR2_TEST_FIXTURE for reference parity")
        }
        let (dit, vae, text) = try SeedVR2Bundle(directory: URL(fileURLWithPath: bundle)).load()
        let expected = try MLX.loadArrays(url: URL(fileURLWithPath: fixture))
        let input = try XCTUnwrap(expected["image"])
        let encoded = vae.encode(input)
        let noise = SeedVR2LatentCreator.noiseLatents(
            seed: 7, height: encoded.dim(-2), width: encoded.dim(-1))
        let modelInput = concatenated([noise, SeedVR2LatentCreator.condition(encoded)], axis: 1)
        let prediction = try dit(modelInput, text, timestep: MLXArray(Float(1000)))
        let decoded = vae.decode(noise - prediction)
        let corrected = try SeedVR2ColorCorrect.labTransfer(
            content: try XCTUnwrap(expected["decoded"]).squeezed(axis: 2), style: input)
        for (name, actual) in [
            ("encoded", encoded), ("noise", noise), ("prediction", prediction),
            ("decoded", decoded), ("corrected", corrected),
        ] {
            let reference = try XCTUnwrap(expected[name])
            XCTAssertEqual(actual.shape, reference.shape, name)
            let delta = abs(actual.asType(.float32) - reference.asType(.float32))
            let maximum = try delta.max().asArrayChecked(Float.self)[0]
            let mean = try delta.mean().asArrayChecked(Float.self)[0]
            print("SEEDVR2_PARITY \(name) max=\(maximum) mean=\(mean)")
            XCTAssertLessThanOrEqual(maximum, name == "noise" ? 0.0000001 : 0.02, name)
        }
    }
    func testEXIFOrientationMatchesReference() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("rotated.png")
        try XCTUnwrap(
            Data(
                base64Encoded:
                    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAADCAIAAAA2iEnWAAAAGmVYSWZNTQAqAAAACAABARIAAwAAAAEABgAAAAAAANZnS2kAAAAbSURBVHicY/jPwMAAwgz///9nYDghEsBltAUASV0HGPblHrcAAAAASUVORK5CYII="
            )
        ).write(to: url)
        let image = try SeedVR2Preprocess.read(url, scale: 2)
        XCTAssertEqual(image.width, 6)
        XCTAssertEqual(image.height, 4)
        let actual = image.tensor[0, 0..., 0 ..< 4, 0 ..< 6].transposed(1, 2, 0)
        let bytes = try MLX.round((actual + 1) * 127.5).asType(.uint8).asArrayChecked(UInt8.self)
        XCTAssertEqual(
            bytes,
            [
                237, 21, 53, 164, 10, 117, 10, 0, 251, 28, 0, 230, 215, 0, 57, 255, 0, 0, 173, 24,
                92, 137, 31, 125, 67, 47, 194, 79, 53, 167, 171, 53, 41, 202, 53, 0, 45, 30, 169,
                81, 74, 140, 178, 167, 80, 179, 202, 44, 85, 202, 11, 53, 202, 0, 0, 33, 208, 54,
                95, 148, 235, 228, 23, 230, 255, 0, 41, 255, 0, 0, 255, 0,
            ])
    }

    func testTransparentInputPreservesReferenceRGB() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("alpha.png")
        try XCTUnwrap(
            Data(
                base64Encoded:
                    "iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAYAAACddGYaAAAAGUlEQVR4nGP4IOLGcCLFqEFy2sP/TAxIAABq5gW6drLNRgAAAABJRU5ErkJggg=="
            )
        ).write(to: url)
        let image = try SeedVR2Preprocess.read(url, scale: 2)
        XCTAssertEqual(image.width, 6)
        XCTAssertEqual(image.height, 4)
        let actual = image.tensor[0, 0..., 0 ..< 4, 0 ..< 6].transposed(1, 2, 0)
        let bytes = try MLX.round((actual + 1) * 127.5).asType(.uint8).asArrayChecked(UInt8.self)
        XCTAssertEqual(
            bytes,
            [
                244, 13, 72, 236, 34, 62, 221, 79, 42, 159, 117, 87, 57, 142, 191, 10, 154, 240,
                244, 13, 72, 236, 34, 62, 221, 79, 42, 159, 117, 87, 57, 142, 191, 10, 154, 240,
                244, 13, 72, 236, 34, 62, 221, 79, 42, 159, 117, 87, 57, 142, 191, 10, 154, 240,
                244, 13, 72, 236, 34, 62, 221, 79, 42, 159, 117, 87, 57, 142, 191, 10, 154, 240,
            ])
    }

    func testResizeMatchesPillow() throws {
        let source = (0 ..< (7 * 5 * 3)).map { UInt8(($0 * 37) % 251) }
        let actual = try SeedVR2Preprocess.resizeRGB(
            source, width: 7, height: 5, targetWidth: 14, targetHeight: 10)
        let expected: [UInt8] = [
            0, 25, 62, 17, 59, 96, 77, 131, 168, 143, 123, 160, 210, 29, 66, 197, 15, 59, 103, 81,
            138, 96, 156, 131, 175, 237, 37, 168, 224, 23, 74, 116, 89, 66, 101, 162, 138, 174, 235,
            172, 209, 255, 2, 31, 68, 23, 65, 102, 83, 137, 174, 150, 129, 166, 221, 35, 72, 208,
            23, 65, 110, 92, 144, 102, 146, 137, 181, 183, 43, 174, 170, 31, 80, 106, 100, 72, 112,
            153, 144, 182, 185, 178, 215, 200, 9, 44, 81, 36, 78, 115, 95, 150, 187, 166, 142, 179,
            244, 48, 85, 231, 40, 78, 126, 116, 157, 114, 126, 150, 194, 67, 56, 187, 54, 48, 93,
            86, 124, 85, 135, 133, 157, 198, 78, 191, 228, 53, 19, 57, 93, 50, 91, 129, 112, 163,
            205, 163, 155, 197, 198, 61, 99, 185, 54, 91, 123, 134, 170, 133, 122, 163, 213, 17, 69,
            205, 4, 62, 107, 82, 142, 98, 154, 130, 170, 217, 32, 204, 246, 0, 32, 69, 104, 65, 103,
            144, 135, 175, 227, 142, 167, 219, 80, 73, 114, 67, 66, 102, 100, 145, 182, 158, 137,
            175, 239, 39, 81, 227, 25, 74, 122, 95, 153, 109, 168, 145, 182, 238, 53, 216, 255, 11,
            44, 81, 122, 78, 115, 141, 151, 187, 181, 138, 179, 172, 33, 85, 110, 21, 78, 120, 102,
            157, 200, 160, 150, 192, 193, 56, 94, 180, 44, 86, 118, 114, 165, 127, 165, 158, 200,
            195, 70, 234, 206, 29, 56, 93, 148, 90, 127, 122, 162, 199, 66, 153, 191, 54, 55, 97,
            87, 47, 90, 145, 127, 169, 226, 137, 162, 214, 75, 68, 109, 62, 60, 97, 95, 136, 177,
            151, 144, 170, 224, 86, 82, 255, 59, 41, 69, 106, 166, 103, 140, 120, 175, 212, 21, 167,
            204, 9, 73, 110, 90, 66, 103, 147, 146, 182, 177, 134, 175, 164, 29, 81, 105, 17, 74,
            115, 98, 154, 191, 153, 142, 183, 179, 44, 95, 193, 0, 54, 82, 119, 178, 116, 153, 136,
            188, 225, 44, 180, 217, 36, 86, 123, 116, 79, 116, 125, 158, 195, 61, 150, 188, 48, 52,
            94, 85, 44, 87, 139, 124, 166, 208, 133, 158, 196, 73, 67, 108, 47, 24, 67, 88, 125,
            184, 122, 159, 143, 194, 231, 55, 186, 223, 49, 92, 129, 129, 85, 122, 115, 164, 201, 7,
            157, 194, 0, 63, 100, 75, 57, 93, 150, 137, 172, 216, 123, 165, 202, 22, 77, 114, 0, 36,
            73,
        ]
        XCTAssertEqual(actual, expected)
    }

    func testSourcePreprocessingMatchesReference() throws {
        let env = ProcessInfo.processInfo.environment
        guard let source = env["SEEDVR2_TEST_SOURCE"],
            let fixture = env["SEEDVR2_TEST_FULL_FIXTURE"]
        else {
            throw XCTSkip("Opt-in full image reference")
        }
        let input = try SeedVR2Preprocess.read(URL(fileURLWithPath: source), scale: 2)
        let expected = try MLX.loadArrays(url: URL(fileURLWithPath: fixture))
        let reference = try XCTUnwrap(expected["preprocessed"])
        XCTAssertEqual(input.tensor.shape, reference.shape)
        let delta = abs(input.tensor - reference)
        let maximum = try delta.max().asArrayChecked(Float.self)[0]
        print("SEEDVR2_PREPROCESS max=\(maximum)")
        XCTAssertEqual(maximum, 0)
    }

    func testRankTransferKeepsStableTies() throws {
        XCTAssertEqual(
            try SeedVR2ColorCorrect.histMatch([2, 1, 1, 3], [10, 20, 30, 40]), [30, 10, 20, 40])
    }

    func testNativeUpscaleAndCancel() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundle = env["SEEDVR2_TEST_BUNDLE"], let source = env["SEEDVR2_TEST_SOURCE"],
            let output = env["SEEDVR2_TEST_OUTPUT"]
        else { throw XCTSkip("Opt-in real upscale proof") }
        let model = try SeedVR2(modelPath: URL(fileURLWithPath: bundle), quantize: nil)
        var request = UpscaleRequest(
            sourceImage: URL(fileURLWithPath: source), scale: 2, steps: 40,
            seed: 7, outputDir: URL(fileURLWithPath: output))
        var completed: URL?
        let start = ContinuousClock.now
        for try await event in model.upscale(request) {
            if case .completed(let url, let seed, let effective) = event {
                completed = url
                XCTAssertEqual(seed, 7)
                XCTAssertEqual(effective?.steps, 1)
                XCTAssertEqual(effective?.scale, 2)
                print("SEEDVR2_COMPLETE \(url.path) elapsed=\(start.duration(to: .now))")
            }
        }
        XCTAssertNotNil(completed)
        let filesBefore = try FileManager.default.contentsOfDirectory(atPath: output)
        let interrupt = FluxInterrupt()
        request.interrupt = interrupt
        var requestedAt: ContinuousClock.Instant?
        var cancelled = false
        for try await event in model.upscale(request) {
            switch event {
            case .step:
                requestedAt = .now
                interrupt.cancel()
            case .cancelled:
                cancelled = true
            case .completed:
                XCTFail("Cancelled upscale must not publish an image")
            default: break
            }
        }
        XCTAssertTrue(cancelled)
        let latency = try XCTUnwrap(requestedAt).duration(to: .now)
        print("SEEDVR2_CANCEL latency=\(latency)")
        XCTAssertLessThan(latency, .seconds(2))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: output), filesBefore)
    }

    func testFourTimesCancellationMemory() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundle = env["SEEDVR2_TEST_BUNDLE"],
            let source = env["SEEDVR2_TEST_CANCEL_1024_SOURCE"],
            let output = env["SEEDVR2_TEST_OUTPUT"]
        else { throw XCTSkip("Opt-in 1024 to 4096 cancellation gate") }
        let model = try SeedVR2(modelPath: URL(fileURLWithPath: bundle), quantize: nil)
        let baseline = Memory.activeMemory
        let interrupt = FluxInterrupt()
        let request = UpscaleRequest(
            sourceImage: URL(fileURLWithPath: source), scale: 4, seed: 7,
            outputDir: URL(fileURLWithPath: output), interrupt: interrupt)
        let files = try FileManager.default.contentsOfDirectory(atPath: output)
        var cancelTime: ContinuousClock.Instant?
        var cancelled = false
        for try await event in model.upscale(request) {
            switch event {
            case .step:
                cancelTime = .now
                interrupt.cancel()
            case .cancelled: cancelled = true
            case .completed: XCTFail("Cancelled job published output")
            default: break
            }
        }
        let latency = try XCTUnwrap(cancelTime).duration(to: .now)
        let after = Memory.activeMemory
        print(
            "SEEDVR2_CANCEL_4X latency=\(latency) active_before=\(baseline) active_after=\(after) cache=\(Memory.cacheMemory)"
        )
        XCTAssertTrue(cancelled)
        XCTAssertLessThan(latency, .seconds(2))
        XCTAssertLessThanOrEqual(after, Int(Double(baseline) * 1.05))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: output), files)
        withExtendedLifetime(model) {}
    }

}
