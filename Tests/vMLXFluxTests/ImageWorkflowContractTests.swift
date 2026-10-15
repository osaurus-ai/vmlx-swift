import Foundation
import XCTest
import vMLXFlux

final class ImageWorkflowContractTests: XCTestCase {
    private func bundle(grid: [Double]? = nil) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        let config: [String: Any] = grid.map { ["sample_sigmas": $0] } ?? [:]
        try JSONSerialization.data(withJSONObject: config).write(to: url.appendingPathComponent("model_index.json"))
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }

    private func assertField(_ field: String, _ body: () throws -> Void, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try body(), file: file, line: line) { error in
            guard case FluxError.invalidField(let actual, _, _) = error else {
                return XCTFail("Expected typed invalidField, got \(error)", file: file, line: line)
            }
            XCTAssertEqual(actual, field, file: file, line: line)
        }
    }

    func testLimitsQueryNeedsOnlyConfigurationAndNeverLoadsModel() async throws {
        let url = try bundle(grid: [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        let engine = FluxEngine()
        let limits = try await engine.limits(for: url, name: "qwen-image-2.1-turbo")
        XCTAssertEqual(limits.fixedSteps, 8)
        XCTAssertEqual(limits.dimensionMultiple, 32)
        XCTAssertNil(limits.maxDimension) // Osaurus' 2048 cap is host admission, not architecture.
        let loaded = await engine.loaded
        XCTAssertNil(loaded)
        var request = ImageGenRequest(prompt: "test", width: 1000, seed: 7, outputDir: url)
        assertField("width") { try limits.validate(request) }
        request.width = 1024
        request.steps = 40
        try limits.validate(request) // The grid, not the requested count, governs Turbo.
    }

    func testUnsupportedOrIgnoredGenerationParametersAreRejected() throws {
        let url = try bundle()
        let limits = try QwenImage21.requestLimits(modelPath: url)
        var request = ImageGenRequest(prompt: "test", seed: 7, outputDir: url)
        request.outputFormat = .jpeg
        assertField("outputFormat") { try limits.validate(request) }
        request.outputFormat = .png
        request.numImages = 2
        assertField("numImages") { try limits.validate(request) }
        request.numImages = 1
        request.guidance = .nan
        assertField("guidance") { try limits.validate(request) }
        request.guidance = 1
        request.negativePrompt = " \n "
        try limits.validate(request) // Same empty-prompt policy as Osaurus.
        request.negativePrompt = "blur"
        assertField("negativePrompt") { try limits.validate(request) }
        request.guidance = 2
        try limits.validate(request)
        request.steps = 1
        assertField("steps") { try limits.validate(request) }
        request.steps = 2
        request.width = Int.max - 31
        request.height = 1024
        assertField("dimensions") { try limits.validate(request) }
    }

    func testEditDefaultsDoNotInventPartialDenoising() throws {
        let url = try bundle()
        let limits = try QwenImage21.requestLimits(modelPath: url)
        var request = ImageEditRequest(prompt: "edit", sourceImage: url, outputDir: url)
        XCTAssertNil(request.strength)
        try limits.validate(request)
        request.strength = 0.75
        assertField("strength") { try limits.validate(request) }
        request.strength = 1
        try limits.validate(request)
        request.mask = url
        assertField("mask") { try limits.validate(request) }
        request.mask = nil
        request.sourceImages = []
        assertField("sourceImages") { try limits.validate(request) }
        request.sourceImages = Array(repeating: url, count: 11)
        assertField("sourceImages") { try limits.validate(request) }
    }

    func testTurboWithoutGridCannotAdvertiseLimits() async throws {
        let url = try bundle()
        do {
            _ = try await FluxEngine().limits(for: url, name: "qwen-image-2.1-turbo")
            XCTFail("Turbo without its schedule must not silently use the base schedule")
        } catch FluxError.localModelIncomplete { }
    }

    func testEffectiveParametersPreserveUInt64SeedThroughJSON() throws {
        let expected = EffectiveImageParameters(
            width: 768, height: 1024, steps: 8, guidance: 1, seed: UInt64.max,
            sampler: "euler", scheduler: "bundle-sigmas", sampleSigmas: [1, 0.5])
        let data = try JSONEncoder().encode(expected)
        XCTAssertEqual(try JSONDecoder().decode(EffectiveImageParameters.self, from: data), expected)
    }
}
