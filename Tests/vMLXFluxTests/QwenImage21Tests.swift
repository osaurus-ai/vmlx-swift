import MLX
import XCTest
@testable import vMLXFlux
@testable import vMLXFluxKit
@testable import vMLXFluxModels

final class QwenImage21Tests: XCTestCase {
    override func setUp() {
        super.setUp()
        VMLXFluxModels.registerAll()
    }

    // MARK: routing

    func testQwenImage21BundlesRouteToTheirOwnClass() {
        XCTAssertEqual(MLXStudioModelStore.canonicalName(for: "Qwen-Image-2.1-mflux-8bit"), "qwen-image-2.1")
        XCTAssertEqual(MLXStudioModelStore.canonicalName(for: "OsaurusAI/Qwen-Image-2.1-mflux-4bit"), "qwen-image-2.1")
        XCTAssertEqual(MLXStudioModelStore.canonicalName(for: "qwen-image-21"), "qwen-image-2.1")
        // Qwen-Image 1.x names must keep their existing routes.
        XCTAssertEqual(MLXStudioModelStore.canonicalName(for: "Qwen-Image-mflux-8bit"), "qwen-image")
        XCTAssertEqual(MLXStudioModelStore.canonicalName(for: "Qwen-Image-Edit-mflux-q8"), "qwen-image-edit")
    }

    func testQwenImage21RegistryDefaultsFollowMflux() throws {
        let entry = try XCTUnwrap(ModelRegistry.lookup(name: "qwen-image-2.1"))
        XCTAssertEqual(entry.kind, .imageGen)
        XCTAssertEqual(entry.defaultSteps, 40)
        XCTAssertEqual(entry.defaultGuidance, 1.0)
    }

    func testProcessorDirectorySatisfiesTokenizerComponent() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("q21-\(UUID().uuidString)/Qwen-Image-2.1-mflux-8bit")
        for dir in ["processor", "transformer", "text_encoder", "vae"] {
            try fm.createDirectory(at: root.appendingPathComponent(dir), withIntermediateDirectories: true)
        }
        try Data("{}".utf8).write(to: root.appendingPathComponent("processor/tokenizer.json"))
        for dir in ["transformer", "text_encoder", "vae"] {
            try save(arrays: ["x": MLXArray([Float(1)])], url: root.appendingPathComponent("\(dir)/0.safetensors"))
        }
        defer { try? fm.removeItem(at: root.deletingLastPathComponent()) }
        let local = try MLXStudioModelStore.inspect(directory: root)
        XCTAssertEqual(local.canonicalName, "qwen-image-2.1")
        XCTAssertTrue(local.components.contains(.tokenizer))
        XCTAssertFalse(local.blockedReasons.contains("missing tokenizer"))
    }

    func testFuzzyLookupResolvesBundleDirectoryNamesToFamilyDefaults() throws {
        let qwen = try XCTUnwrap(ModelRegistry.lookupFuzzy(name: "Qwen-Image-2.1-mflux-8bit"))
        XCTAssertEqual(qwen.name, "qwen-image-2.1")
        XCTAssertEqual(qwen.defaultSteps, 40)
        XCTAssertEqual(qwen.defaultGuidance, 1.0)
        XCTAssertEqual(ModelRegistry.lookupFuzzy(name: "OsaurusAI/ideogram-4-mflux-q4")?.name, "ideogram")
        XCTAssertEqual(ModelRegistry.lookupFuzzy(name: "Qwen-Image-mflux-8bit")?.name, "qwen-image")
    }

    // MARK: layout

    func testTextToImageLayoutIsTextPrefixThenTarget() throws {
        let layout = try QwenImage21Layout.create(
            slots: [Bool](repeating: false, count: 44), shapes: [(1, 32, 32)], axes: [16, 56, 56])
        XCTAssertEqual(layout.length, 44 + 1024)
        XCTAssertEqual(layout.prefixLength, 44)
        XCTAssertEqual(layout.targetTokens, 1024)
        XCTAssertEqual(layout.segments.count, 1)
        XCTAssertEqual(layout.segments[0].start, 0)
        XCTAssertEqual(layout.segments[0].end, 44)
        XCTAssertTrue(layout.segments[0].isText)
        XCTAssertEqual(Array(layout.gatherRows[0 ..< 44]), (0 ..< 44).map(Int32.init))
        XCTAssertEqual(Array(layout.gatherRows[44...]), (44 ..< 1068).map(Int32.init))
        XCTAssertEqual(layout.targetMask.filter { $0 }.count, 1024)
        XCTAssertEqual(layout.cos.shape, [1068, 64])
    }

    func testReferenceLayoutExpandsEachVisionSlotToFourLatentTokens() throws {
        // 8 text rows, 4 vision placeholders (a 4x4 latent reference), 3 trailing text rows,
        // then a 2x2 target.
        let slots = [Bool](repeating: false, count: 8) + [Bool](repeating: true, count: 4)
            + [Bool](repeating: false, count: 3)
        let layout = try QwenImage21Layout.create(slots: slots, shapes: [(1, 4, 4), (1, 2, 2)], axes: [16, 56, 56])
        XCTAssertEqual(layout.length, 8 + 16 + 3 + 4)
        XCTAssertEqual(layout.prefixLength, 27)
        XCTAssertEqual(layout.segments.map(\.start), [0, 8, 24])
        XCTAssertEqual(layout.segments.map(\.end), [8, 24, 27])
        XCTAssertEqual(layout.segments.map(\.isText), [true, false, true])
        // reference rows read image rows T+0..T+15, trailing text reads text rows 12..14
        XCTAssertEqual(Array(layout.gatherRows[8 ..< 24]), (15 ..< 31).map(Int32.init))
        XCTAssertEqual(Array(layout.gatherRows[24 ..< 27]), [12, 13, 14])
        XCTAssertEqual(Array(layout.gatherRows[27...]), (31 ..< 35).map(Int32.init))
    }

    func testLayoutRejectsMismatchedReferenceSlots() {
        XCTAssertThrowsError(try QwenImage21Layout.create(
            slots: [false, true], shapes: [(1, 4, 4), (1, 2, 2)], axes: [16, 56, 56]))
    }

    // MARK: schedule

    func testScheduleMatchesMfluxLinearSchedulerAt512() {
        // mflux main 03700b9, 8 steps, 512x512 (dumped from LinearScheduler).
        let expected: [Float] = [
            1.0, 0.9061342477798462, 0.8013588190078735, 0.6836543083190918,
            0.550470232963562, 0.3985380530357361, 0.2235991358757019, 0.020000040531158447, 0.0,
        ]
        let sigmas = QwenImage21Schedule.sigmas(steps: 8, width: 512, height: 512).asArray(Float.self)
        XCTAssertEqual(sigmas.count, expected.count)
        for (a, b) in zip(sigmas, expected) { XCTAssertEqual(a, b, accuracy: 1e-6) }
    }

    // MARK: mflux-saved key layout

    func testTextEncoderLookupFallsBackToPrefixFreeMfluxKeys() {
        let loaded = LoadedWeights(
            weights: [:],
            componentWeights: ["text_encoder": ["embed_tokens.weight": MLXArray([Float(1)])]])
        let store = MFluxStore(loaded)
        XCTAssertTrue(store.hasKey("text_encoder", "language_model.embed_tokens.weight"))
        XCTAssertFalse(store.hasKey("transformer", "language_model.embed_tokens.weight"))
    }

    // MARK: Turbo

    func testTurboBundlesRouteToTheTurboEntryBeforeQwenImage21() {
        for name in [
            "Qwen-Image-2.1-Turbo-mflux-8bit", "OsaurusAI/Qwen-Image-2.1-Turbo-mflux-4bit",
            "qwen-image-21-turbo", "Qwen-Image-2.1-Turbo",
        ] {
            XCTAssertEqual(MLXStudioModelStore.canonicalName(for: name), "qwen-image-2.1-turbo", name)
        }
        XCTAssertEqual(MLXStudioModelStore.canonicalName(for: "Qwen-Image-2.1-mflux-8bit"), "qwen-image-2.1")
        // Z-Image-Turbo is not a Qwen-Image-2.1 bundle.
        XCTAssertEqual(MLXStudioModelStore.canonicalName(for: "Z-Image-Turbo-mflux-4bit"), "z-image-turbo")
    }

    func testTurboRegistryDefaultsAreEightStepsWithoutCFG() throws {
        _ = QwenImage21._register
        let entry = try XCTUnwrap(ModelRegistry.lookup(name: "qwen-image-2.1-turbo"))
        XCTAssertEqual(entry.kind, .imageGen)
        XCTAssertEqual(entry.defaultSteps, 8)
        XCTAssertEqual(entry.defaultGuidance, 1.0)
        let fuzzy = try XCTUnwrap(ModelRegistry.lookupFuzzy(name: "Qwen-Image-2.1-Turbo-mflux-6bit"))
        XCTAssertEqual(fuzzy.name, "qwen-image-2.1-turbo")
        XCTAssertEqual(fuzzy.defaultSteps, 8)
    }

    /// The Turbo bundle's model_index.json carries `sample_sigmas`; they are used unshifted with a
    /// terminal 0 (diffusers FlowMatchEulerDiscreteScheduler with shift 1.0, no dynamic or terminal shift).
    func testSampleSigmasFromModelIndexAreUsedUnshiftedWithTerminalZero() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("q21-turbo-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let grid: [Double] = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
        let index: [String: Any] = ["_class_name": "QwenImage21Pipeline", "sample_sigmas": grid]
        try JSONSerialization.data(withJSONObject: index).write(to: dir.appendingPathComponent("model_index.json"))
        let parsed = try XCTUnwrap(try QwenImage21Schedule.sampleSigmas(modelPath: dir))
        XCTAssertEqual(parsed.count, 8)
        let sigmas = QwenImage21Schedule.sigmas(grid: parsed).asArray(Float.self)
        XCTAssertEqual(sigmas.count, 9)
        for (a, b) in zip(sigmas, grid + [0]) { XCTAssertEqual(Double(a), b, accuracy: 1e-6) }
        // A Qwen-Image-2.1 bundle (no grid) keeps the shifted linear schedule.
        let plain = FileManager.default.temporaryDirectory
            .appendingPathComponent("q21-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: plain, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: ["_class_name": "QwenImage21Pipeline"])
            .write(to: plain.appendingPathComponent("model_index.json"))
        XCTAssertNil(try QwenImage21Schedule.sampleSigmas(modelPath: plain))
    }
    func testMalformedFixedGridFailsInsteadOfUsingTheBaseSchedule() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        for raw in ["null", "[]", "[1]", "[true, false]", "[1, 0.5, 0.7]", "[1, 1]",
                    "[1, 0]", "[1, -0.2]", "[2, 0.5]", "[1, 1e100]", "[1, \"bad\"]"] {
            try Data("{\"sample_sigmas\":\(raw)}".utf8).write(to: dir.appendingPathComponent("model_index.json"))
            XCTAssertThrowsError(try QwenImage21Schedule.sampleSigmas(modelPath: dir), raw)
        }
    }

    func testTurboLoaderRejectsMissingGridBeforeLoadingWeights() async throws {
        _ = QwenImage21._register
        let entry = try XCTUnwrap(ModelRegistry.lookup(name: "qwen-image-2.1-turbo"))
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            _ = try await entry.loader(dir, 4)
            XCTFail("Turbo must not fall back to the base schedule")
        } catch {
            XCTAssertTrue(String(describing: error).contains("sample_sigmas"), "\(error)")
        }
    }

}
