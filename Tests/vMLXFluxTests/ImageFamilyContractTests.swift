import Foundation
import XCTest
import vMLXFluxKit
import vMLXFluxModels

final class ImageFamilyContractTests: XCTestCase {
    func testImplementedFamilyLimitsDoNotReadWeights() throws {
        VMLXFluxModels.registerAll()
        let missing = URL(fileURLWithPath: "/does-not-exist/weights")
        let start = Date()
        for name in ["flux1-schnell", "z-image-turbo", "qwen-image", "qwen-image-edit", "ideogram"] {
            let entry = try XCTUnwrap(ModelRegistry.lookup(name: name))
            let query = try XCTUnwrap(entry.limits, name)
            let limits = try query(missing)
            XCTAssertTrue(limits.supportedFormats == [.png])
            if name != "qwen-image-edit" {
                var request = ImageGenRequest(prompt: "cube", width: 1000, height: 1000, outputDir: missing)
                XCTAssertThrowsError(try limits.validate(request), name)
                request.width = 512; request.height = 512
                try limits.validate(request)
                request.outputFormat = .jpeg
                XCTAssertThrowsError(try limits.validate(request), name)
            }
        }
        XCTAssertLessThan(Date().timeIntervalSince(start), 0.1)
    }

    func testGuidanceSemanticsMatchImplementedNegativeConditioning() throws {
        VMLXFluxModels.registerAll()
        let root = FileManager.default.temporaryDirectory
        let z = try XCTUnwrap(ModelRegistry.lookup(name: "z-image-turbo")?.limits)(root)
        let qwen = try XCTUnwrap(ModelRegistry.lookup(name: "qwen-image")?.limits)(root)
        let flux = try XCTUnwrap(ModelRegistry.lookup(name: "flux1-schnell")?.limits)(root)
        var request = ImageGenRequest(prompt: "cube", negativePrompt: "blue", guidance: 0.5, outputDir: root)
        try z.validate(request) // Z-Image uses conditional + guidance*(conditional-negative).
        try qwen.validate(request) // Qwen always executes both branches.
        XCTAssertThrowsError(try flux.validate(request))
        request.guidance = 0
        XCTAssertThrowsError(try z.validate(request))
        try qwen.validate(request)
        XCTAssertEqual(try JSONDecoder().decode(FluxModelLimits.self, from: JSONEncoder().encode(qwen)), qwen)
    }

    func testEditRejectsUnsupportedMaskAndPreservesRounding() throws {
        VMLXFluxModels.registerAll()
        let root = FileManager.default.temporaryDirectory
        let limits = try XCTUnwrap(ModelRegistry.lookup(name: "qwen-image-edit")?.limits)(root)
        var request = try ImageEditRequest(prompt: "cube", sourceImages: [root.appendingPathComponent("source.png")], outputDir: root)
        try limits.validate(request)
        request.mask = root.appendingPathComponent("mask.png")
        XCTAssertThrowsError(try limits.validate(request))
        request.mask = nil
        request.width = 1000; request.height = 1000
        try limits.validate(request) // Existing edit floor-to16 behavior remains supported.
    }

    func testOldLimitsPayloadDecodesWithSingleImageDefaults() throws {
        let bytes = Data(#"{"dimensionMultiple":16,"minDimension":16,"minimumSteps":1,"supportsNegativePrompt":false,"supportsEditNegativePrompt":false,"supportsStrength":false,"supportsMask":false,"maxSourceImages":0,"supportedFormats":["png"]}"#.utf8)
        let limits = try JSONDecoder().decode(FluxModelLimits.self, from: bytes)
        XCTAssertEqual(limits.maximumImages, 1)
        XCTAssertEqual(limits.maximumParallelImages, 1)
    }

    func testRealFamilyMetadataReplay() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let name = env["VMLX_FLUX_FAMILY_NAME"], let path = env["VMLX_FLUX_FAMILY_MODEL"] else {
            throw XCTSkip("Set VMLX_FLUX_FAMILY_NAME and VMLX_FLUX_FAMILY_MODEL for real image replay")
        }
        VMLXFluxModels.registerAll()
        let entry = try XCTUnwrap(ModelRegistry.lookup(name: name))
        let model = try await entry.loader(URL(fileURLWithPath: path), nil)
        let generator = try XCTUnwrap(model as? ImageGenerator)
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("flux-family-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: out) }
        func generate(_ request: ImageGenRequest) async throws -> (Data, EffectiveImageParameters) {
            var output: (URL, EffectiveImageParameters)?
            var steps = 0
            for try await event in generator.generate(request) {
                switch event {
                case .step: steps += 1
                case .completed(let url, let seed, let effective):
                    let parameters = try XCTUnwrap(effective)
                    XCTAssertEqual(parameters.seed, seed)
                    let size = try ImageIO.dimensions(of: url)
                    XCTAssertEqual(size.width, parameters.width); XCTAssertEqual(size.height, parameters.height)
                    output = (url, parameters)
                case .failed(let message, _): XCTFail(message)
                default: break
                }
            }
            let value = try XCTUnwrap(output)
            XCTAssertEqual(steps, value.1.steps)
            return (try Data(contentsOf: value.0), value.1)
        }
        let original = try await generate(ImageGenRequest(prompt: "A red cube on a white table",
            width: 512, height: 512, steps: entry.defaultSteps, guidance: entry.defaultGuidance,
            seed: 7, outputDir: out))
        let p = original.1
        let replay = try await generate(ImageGenRequest(prompt: "A red cube on a white table",
            width: p.width, height: p.height, steps: p.steps, guidance: p.guidance, seed: p.seed, outputDir: out))
        XCTAssertEqual(original.0, replay.0)
        print("FLUX_FAMILY_REPLAY name=\(name) width=\(p.width) height=\(p.height) steps=\(p.steps) seed=\(p.seed) bytes=\(original.0.count)")
    }

    func testRealEditRoundedMetadataReplay() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["VMLX_FLUX_EDIT_MODEL"], let source = env["VMLX_FLUX_EDIT_SOURCE"] else {
            throw XCTSkip("Set VMLX_FLUX_EDIT_MODEL and VMLX_FLUX_EDIT_SOURCE for real edit replay")
        }
        VMLXFluxModels.registerAll()
        let entry = try XCTUnwrap(ModelRegistry.lookup(name: "qwen-image-edit"))
        let model = try await entry.loader(URL(fileURLWithPath: path), nil)
        let editor = try XCTUnwrap(model as? ImageEditor)
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("flux-edit-replay-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: out) }
        func edit(_ request: ImageEditRequest) async throws -> (Data, EffectiveImageParameters) {
            var output: (URL, EffectiveImageParameters)?
            var steps = 0
            for try await event in editor.edit(request) {
                switch event {
                case .step: steps += 1
                case .completed(let url, let seed, let effective):
                    let parameters = try XCTUnwrap(effective)
                    let size = try ImageIO.dimensions(of: url)
                    XCTAssertEqual(parameters.seed, seed)
                    XCTAssertEqual(parameters.width, size.width)
                    XCTAssertEqual(parameters.height, size.height)
                    output = (url, parameters)
                case .failed(let message, _): XCTFail(message)
                default: break
                }
            }
            let result = try XCTUnwrap(output)
            XCTAssertEqual(result.1.steps, steps)
            return (try Data(contentsOf: result.0), result.1)
        }
        let original = try await edit(ImageEditRequest(prompt: "Change the red cube to blue",
            sourceImage: URL(fileURLWithPath: source), width: 1000, height: 1000,
            steps: entry.defaultSteps, guidance: entry.defaultGuidance, seed: 7, outputDir: out))
        let p = original.1
        XCTAssertEqual(p.width, 992)
        XCTAssertEqual(p.height, 992)
        let replay = try await edit(ImageEditRequest(prompt: "Change the red cube to blue",
            sourceImage: URL(fileURLWithPath: source), width: p.width, height: p.height,
            steps: p.steps, guidance: p.guidance, seed: p.seed, outputDir: out))
        XCTAssertEqual(original.0, replay.0)
        print("FLUX_EDIT_REPLAY width=\(p.width) height=\(p.height) steps=\(p.steps) seed=\(p.seed) bytes=\(original.0.count)")
    }
}
