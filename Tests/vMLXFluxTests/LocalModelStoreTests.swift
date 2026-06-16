import XCTest
@testable import vMLXFlux
@testable import vMLXFluxKit
@testable import vMLXFluxModels

final class LocalModelStoreTests: XCTestCase {

    override func setUp() {
        super.setUp()
        VMLXFluxModels.registerAll()
    }

    func testInspectRecognizesMFluxComponentLayout() throws {
        let root = try makeTemporaryImageModelRoot()
        let model = root.appendingPathComponent("Z-Image-Turbo-mflux-4bit", isDirectory: true)
        try makeComponentLayout(at: model)

        let inspected = try MLXStudioModelStore.inspect(directory: model)

        XCTAssertEqual(inspected.canonicalName, "z-image-turbo")
        XCTAssertEqual(inspected.quantizationBits, 4)
        XCTAssertEqual(inspected.readiness, .loadableScaffold)
        XCTAssertTrue(inspected.components.contains(.tokenizer))
        XCTAssertTrue(inspected.components.contains(.transformer))
        XCTAssertTrue(inspected.components.contains(.textEncoder))
        XCTAssertTrue(inspected.components.contains(.vae))
        XCTAssertEqual(inspected.safetensorCount, 3)
    }

    func testStoreResolveUsesCanonicalAndDirectoryNames() throws {
        let root = try makeTemporaryImageModelRoot()
        let model = root.appendingPathComponent("qwen-image-mflux-4bit", isDirectory: true)
        try makeComponentLayout(at: model)

        let store = MLXStudioModelStore(root: root)

        let byCanonical = try store.resolve(name: "qwen-image")
        let byDirectory = try store.resolve(name: "qwen-image-mflux-4bit")

        // Compare bundle identity (last path component) rather than the full
        // absolute URL: the store canonicalizes the macOS /var -> /private/var
        // temp-dir symlink, so absolute-URL equality is brittle.
        XCTAssertEqual(byCanonical?.directory.lastPathComponent, model.lastPathComponent)
        XCTAssertEqual(byCanonical?.canonicalName, "qwen-image")
        XCTAssertEqual(byDirectory?.directory.lastPathComponent, model.lastPathComponent)
        XCTAssertEqual(byDirectory?.canonicalName, "qwen-image")
    }

    func testScanExpandsNestedQuantVariantBundles() throws {
        let root = try makeTemporaryImageModelRoot()
        let model = root.appendingPathComponent("Qwen-Image-Edit-mflux", isDirectory: true)
        let q4 = model.appendingPathComponent("q4", isDirectory: true)
        try makeComponentLayout(at: q4)

        let scanned = try MLXStudioModelStore(root: root).scan()
        let variant = try XCTUnwrap(scanned.first {
            $0.directoryName == "Qwen-Image-Edit-mflux-q4"
        })

        XCTAssertEqual(variant.directory.lastPathComponent, "q4")
        XCTAssertEqual(variant.canonicalName, "qwen-image-edit")
        XCTAssertEqual(variant.quantizationBits, 4)
        XCTAssertEqual(variant.readiness, .loadableScaffold)
        XCTAssertTrue(variant.components.contains(.tokenizer))
        XCTAssertTrue(variant.components.contains(.transformer))
        XCTAssertTrue(variant.components.contains(.textEncoder))
        XCTAssertTrue(variant.components.contains(.vae))
    }

    func testIdeogramRequiresUnconditionalTransformerComponent() throws {
        let root = try makeTemporaryImageModelRoot()
        let model = root.appendingPathComponent("ideogram-4-nf4", isDirectory: true)
        try makeComponentLayout(at: model)

        let incomplete = try MLXStudioModelStore.inspect(directory: model)

        XCTAssertEqual(incomplete.canonicalName, "ideogram")
        XCTAssertEqual(incomplete.readiness, .incomplete)
        XCTAssertTrue(incomplete.blockedReasons.contains("missing unconditionalTransformer"))

        let unconditional = model.appendingPathComponent("unconditional_transformer", isDirectory: true)
        try FileManager.default.createDirectory(at: unconditional, withIntermediateDirectories: true)
        try Data([3]).write(to: unconditional.appendingPathComponent("0.safetensors"))

        let complete = try MLXStudioModelStore.inspect(directory: model)

        XCTAssertEqual(complete.readiness, .loadableScaffold)
        XCTAssertTrue(complete.components.contains(.unconditionalTransformer))
    }

    func testEngineLoadFromStoreRejectsIncompleteLocalBundleBeforeWeightLoad() async throws {
        let root = try makeTemporaryImageModelRoot()
        let model = root.appendingPathComponent("Z-Image-Turbo-mflux-4bit", isDirectory: true)
        try FileManager.default.createDirectory(
            at: model.appendingPathComponent("tokenizer", isDirectory: true),
            withIntermediateDirectories: true)

        let engine = FluxEngine()
        let store = MLXStudioModelStore(root: root)

        do {
            _ = try await engine.load(name: "z-image-turbo", from: store)
            XCTFail("expected incomplete local model rejection")
        } catch FluxError.localModelIncomplete(let url, let reasons) {
            XCTAssertEqual(url.lastPathComponent, model.lastPathComponent)
            XCTAssertTrue(reasons.contains("no safetensors found"))
            XCTAssertTrue(reasons.contains("missing transformer"))
        } catch {
            XCTFail("wrong error: \(error)")
        }
    }

    private func makeTemporaryImageModelRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx-flux-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: root)
        }
        return root
    }

    private func makeComponentLayout(at model: URL) throws {
        let fm = FileManager.default
        for component in ["tokenizer", "transformer", "text_encoder", "vae"] {
            try fm.createDirectory(
                at: model.appendingPathComponent(component, isDirectory: true),
                withIntermediateDirectories: true)
        }
        try Data("{}".utf8).write(
            to: model.appendingPathComponent("tokenizer/tokenizer.json"))
        try Data([0]).write(
            to: model.appendingPathComponent("transformer/0.safetensors"))
        try Data([1]).write(
            to: model.appendingPathComponent("text_encoder/0.safetensors"))
        try Data([2]).write(
            to: model.appendingPathComponent("vae/0.safetensors"))
    }
}
