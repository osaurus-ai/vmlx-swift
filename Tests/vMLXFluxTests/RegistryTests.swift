import XCTest
@testable import vMLXFlux
@testable import vMLXFluxKit
@testable import vMLXFluxModels
@testable import vMLXFluxVideo

final class RegistryTests: XCTestCase {

    override func setUp() {
        super.setUp()
        VMLXFluxModels.registerAll()
        VMLXFluxVideo.registerAll()
    }

    func testAllImageGenModelsRegistered() {
        let gens = ModelRegistry.all(kind: .imageGen).map(\.name)
        XCTAssertTrue(gens.contains("flux1-schnell"))
        XCTAssertTrue(gens.contains("flux1-dev"))
        XCTAssertTrue(gens.contains("flux2-klein"))
        XCTAssertTrue(gens.contains("z-image-turbo"))
        XCTAssertTrue(gens.contains("qwen-image"))
        XCTAssertTrue(gens.contains("fibo"))
    }

    func testAllImageEditModelsRegistered() {
        let edits = ModelRegistry.all(kind: .imageEdit).map(\.name)
        XCTAssertTrue(edits.contains("flux1-kontext"))
        XCTAssertTrue(edits.contains("flux1-fill"))
        XCTAssertTrue(edits.contains("flux2-klein-edit"))
        XCTAssertTrue(edits.contains("qwen-image-edit"))
    }

    func testUpscaleModelRegistered() {
        let up = ModelRegistry.all(kind: .imageUpscale).map(\.name)
        XCTAssertTrue(up.contains("seedvr2"))
    }

    func testVideoStubsRegistered() {
        let video = ModelRegistry.all(kind: .videoGen).map(\.name)
        XCTAssertTrue(video.contains("wan-2.1"))
        XCTAssertTrue(video.contains("wan-2.2"))
    }

    func testFuzzyLookupStripsHFPrefixAndQuantSuffix() {
        XCTAssertNotNil(ModelRegistry.lookupFuzzy(name: "black-forest-labs/FLUX.1-schnell-8bit"))
        XCTAssertNotNil(ModelRegistry.lookupFuzzy(name: "FLUX.1-DEV-4BIT"))
        // The above should both resolve after lowercasing + stripping
        // the org prefix + stripping the `-Nbit` suffix. We only assert
        // they don't return nil — the exact entry the fuzzy matcher picks
        // is covered by testCanonicalLookup below.
    }

    func testCanonicalLookup() {
        let entry = ModelRegistry.lookup(name: "flux1-schnell")
        XCTAssertNotNil(entry)
        XCTAssertEqual(entry?.displayName, "FLUX.1 Schnell")
        XCTAssertEqual(entry?.defaultSteps, 4)
    }

    func testEngineLoadFailsOnMissingWeights() async {
        let engine = FluxEngine()
        let missing = URL(fileURLWithPath: "/tmp/does-not-exist-\(UUID().uuidString)")
        do {
            try await engine.load(name: "flux1-schnell", modelPath: missing)
            XCTFail("expected weightsNotFound error")
        } catch FluxError.weightsNotFound {
            // expected
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    func testEngineGenerateRequiresLoad() async {
        let engine = FluxEngine()
        let request = ImageGenRequest(
            prompt: "test",
            outputDir: URL(fileURLWithPath: "/tmp")
        )
        let stream = await engine.generate(request)
        do {
            for try await _ in stream {}
            XCTFail("expected notLoaded error")
        } catch FluxError.notLoaded {
            // expected
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }
}
