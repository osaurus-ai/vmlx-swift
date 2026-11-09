import Foundation
import XCTest
import vMLXFluxKit
import vMLXFluxModels

final class ZImageTurboContractTests: XCTestCase {
    // The installed quant declares mflux 0.18.0. At its v.0.18.0 tag
    // (48e5ae662c003db697af733f677885739483ff28), ModelConfig.z_image_turbo
    // has supports_guidance=false; ZImage.generate_image resolves CFG to zero.
    // The publisher likewise distinguishes Turbo (no CFG) from base Z-Image.
    func testTurboAdvertisesNoNegativeConditioning() throws {
        VMLXFluxModels.registerAll()
        let entry = try XCTUnwrap(ModelRegistry.lookup(name: "z-image-turbo"))
        let limits = try XCTUnwrap(entry.limits)(URL(fileURLWithPath: "/missing-public-model"))
        XCTAssertFalse(limits.supportsNegativePrompt)
        XCTAssertTrue(limits.ignoresNegativePrompt)
    }

    func testLegacyRequestDefaultsAndIgnoredOptionsRemainAdmitted() throws {
        VMLXFluxModels.registerAll()
        let entry = try XCTUnwrap(ModelRegistry.lookup(name: "z-image-turbo"))
        let missing = URL(fileURLWithPath: "/missing-public-model")
        let limits = try XCTUnwrap(entry.limits)(missing)
        // The generic library initializer carries 3.5; rejecting it would
        // break existing callers that omitted guidance. Turbo resolves its
        // own documented effective CFG, rather than adding a new default.
        var request = ImageGenRequest(prompt: "cube", outputDir: missing)
        try limits.validate(request)
        for guidance: Float in [0, 0.5, 1, 7] {
            request.guidance = guidance
            request.negativePrompt = "text, watermark"
            try limits.validate(request)
        }
        XCTAssertEqual(entry.defaultGuidance, 0)
        XCTAssertEqual(entry.defaultSteps, 4) // Existing step default is a separate audit.
        request.guidance = .nan
        XCTAssertThrowsError(try limits.validate(request))
    }

    func testOtherFamilyNegativeContractsStayDistinct() throws {
        VMLXFluxModels.registerAll()
        let missing = URL(fileURLWithPath: "/missing-public-model")
        func limits(_ name: String) throws -> FluxModelLimits {
            try XCTUnwrap(ModelRegistry.lookup(name: name)?.limits)(missing)
        }
        let request = ImageGenRequest(prompt: "cube", negativePrompt: "blue",
            guidance: 0.5, outputDir: missing)
        try limits("qwen-image").validate(request)
        XCTAssertThrowsError(try limits("flux1-schnell").validate(request))
        XCTAssertThrowsError(try limits("ideogram").validate(request))
        XCTAssertFalse(try limits("qwen-image-edit").supportsEditNegativePrompt)
    }
}
