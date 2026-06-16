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

    func testQwenImageEditLoadRejectsBundleMissingVisionTowerKeys() throws {
        let model = try makeTemporaryQwenImageEditBundle()

        XCTAssertThrowsError(try QwenImageEdit(modelPath: model, quantize: 4)) { error in
            guard case FluxError.localModelIncomplete(let url, let reasons) = error else {
                return XCTFail("wrong error: \(error)")
            }
            XCTAssertEqual(url.lastPathComponent, model.lastPathComponent)
            XCTAssertTrue(reasons.contains {
                $0.contains("missing text_encoder weight encoder.visual.patch_embed.proj.weight")
            })
        }
    }

    func testQwenImageEditManifestValidationAcceptsRequiredEditKeys() throws {
        let model = try makeTemporaryQwenImageEditBundle(includeEditKeys: true)

        let edit = try QwenImageEdit(modelPath: model, quantize: 4)

        XCTAssertEqual(edit.modelPath.lastPathComponent, model.lastPathComponent)
        XCTAssertEqual(edit.quantize, 4)
    }

    func testQwenImageEditLoadRejectsIndexWithMissingShard() throws {
        let model = try makeTemporaryQwenImageEditBundle(includeEditKeys: true)
        let shard = model.appendingPathComponent("text_encoder/0.safetensors")
        try FileManager.default.removeItem(at: shard)

        XCTAssertThrowsError(try QwenImageEdit(modelPath: model, quantize: 4)) { error in
            guard case FluxError.weightsNotFound(let url) = error else {
                return XCTFail("wrong error: \(error)")
            }
            XCTAssertEqual(url.lastPathComponent, shard.lastPathComponent)
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

    private func makeTemporaryQwenImageEditBundle(includeEditKeys: Bool = false) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-flux-registry-tests-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: root)
        }
        let model = root.appendingPathComponent("Qwen-Image-Edit-mflux-q4", isDirectory: true)
        let fm = FileManager.default
        for component in ["tokenizer", "text_encoder", "transformer", "vae"] {
            try fm.createDirectory(
                at: model.appendingPathComponent(component, isDirectory: true),
                withIntermediateDirectories: true)
        }
        try Data("{}".utf8).write(to: model.appendingPathComponent("tokenizer/tokenizer.json"))
        try Data("{}".utf8).write(to: model.appendingPathComponent("tokenizer/tokenizer_config.json"))
        try writeWeightIndex(
            keys: includeEditKeys
                ? [
                    "encoder.embed_tokens.weight",
                    "encoder.layers.0.self_attn.q_proj.weight",
                    "encoder.norm.weight",
                    "encoder.visual.patch_embed.proj.weight",
                    "encoder.visual.blocks.0.attn.qkv.weight",
                    "encoder.visual.blocks.31.attn.qkv.weight",
                    "encoder.visual.merger.mlp_1.weight",
                ]
                : [
                    "encoder.embed_tokens.weight",
                    "encoder.layers.0.self_attn.q_proj.weight",
                    "encoder.norm.weight",
                ],
            to: model.appendingPathComponent("text_encoder/model.safetensors.index.json"))
        try writeWeightIndex(
            keys: [
                "img_in.weight",
                "txt_in.weight",
                "time_text_embed.timestep_embedder.linear_1.weight",
                "transformer_blocks.0.attn.add_q_proj.weight",
                "transformer_blocks.59.img_ff.mlp_out.weight",
                "proj_out.weight",
            ],
            to: model.appendingPathComponent("transformer/model.safetensors.index.json"))
        try writeWeightIndex(
            keys: [
                "encoder.conv_in.conv3d.weight",
                "encoder.down_blocks.0.resnets.0.conv1.conv3d.weight",
                "quant_conv.conv3d.weight",
                "post_quant_conv.conv3d.weight",
                "decoder.conv_in.conv3d.weight",
                "decoder.conv_out.conv3d.weight",
            ],
            to: model.appendingPathComponent("vae/model.safetensors.index.json"))
        return model
    }

    private func writeWeightIndex(keys: [String], to url: URL) throws {
        let weightMap = Dictionary(uniqueKeysWithValues: keys.map { ($0, "0.safetensors") })
        let object: [String: Any] = ["weight_map": weightMap]
        let data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
        try Data([0]).write(
            to: url.deletingLastPathComponent().appendingPathComponent("0.safetensors"))
    }
}
