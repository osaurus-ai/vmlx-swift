import Foundation
import XCTest
@testable import vMLXFluxKit
@testable import vMLXFluxModels

final class Flux3KleinTests: XCTestCase {
    private func fixture(_ body: (URL) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("flux3-\(UUID().uuidString)/FLUX-3-Image-Klein-mflux-4bit")
        let fm = FileManager.default
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root.deletingLastPathComponent()) }
        let source = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Resources/Flux3/model_index.json")
        try fm.copyItem(at: source, to: root.appendingPathComponent("model_index.json"))
        for component in ["transformer", "text_encoder", "vae", "scheduler"] {
            try fm.createDirectory(at: root.appendingPathComponent(component), withIntermediateDirectories: true)
        }
        func write(_ file: String, _ text: String) throws { try Data(text.utf8).write(to: root.appendingPathComponent(file)) }
        for component in ["transformer", "text_encoder", "vae"] {
            let sentinels = [
                "transformer": ["target_timestep_embedder.linear_1.weight", "mode_blocks.image.4.ff.linear_out.weight", "single_transformer_blocks.27.attn.to_q.weight"],
                "text_encoder": ["embed_tokens.weight", "layers.31.self_attn.q_proj.weight"],
                "vae": ["bn.running_mean", "bn.running_var", "quant_conv.weight", "post_quant_conv.weight"],
            ]
            let index = ["weight_map": Dictionary(uniqueKeysWithValues: sentinels[component]!.map { ($0, "0.safetensors") })]
            try JSONSerialization.data(withJSONObject: index).write(to: root.appendingPathComponent("\(component)/model.safetensors.index.json"))
            try write("\(component)/0.safetensors", "") // Metadata preflight must never open these files.
            try fm.copyItem(at: source.deletingLastPathComponent().appendingPathComponent("\(component)/config.json"),
                to: root.appendingPathComponent("\(component)/config.json"))
        }
        for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] { try write("text_encoder/\(name)", "{}") }
        try write("scheduler/scheduler_config.json", #"{"_class_name":"Flux3FlowMapScheduler","num_inference_steps":16,"alpha":6.93,"flow_map":true}"#)
        try body(root)
    }

    func testRoutingAndRegistryExcludeBase() throws {
        ModelRegistry.register(Flux3Registration.entry { _, _ in throw FluxError.notImplemented("metadata test loader is never invoked") })
        for bits in [4, 6, 8] {
            let name = "OsaurusAI/FLUX-3-Image-Klein-mflux-\(bits)bit"
            XCTAssertEqual(MLXStudioModelStore.canonicalName(for: name), "flux3-image-klein")
            XCTAssertEqual(ModelRegistry.lookupFuzzy(name: name)?.defaultSteps, 16)
            XCTAssertEqual(ModelRegistry.lookupFuzzy(name: name)?.defaultGuidance, 1)
        }
        XCTAssertEqual(MLXStudioModelStore.canonicalName(for: "flux3-image-klein"), "flux3-image-klein")
        XCTAssertNil(MLXStudioModelStore.canonicalName(for: "FLUX-3-Image-Klein-Base-mflux-4bit"))
        XCTAssertEqual(MLXStudioModelStore.canonicalName(for: "flux2-klein"), "flux2-klein")
        XCTAssertEqual(MLXStudioModelStore.canonicalName(for: "flux1-schnell"), "flux1-schnell")
    }

    func testStampAndLimits() throws {
        try fixture { root in
            let bundle = try Flux3Bundle(modelPath: root)
            XCTAssertEqual(bundle.stamp.defaults.steps, 16)
            XCTAssertEqual(bundle.stamp.defaults.width, 1024)
            XCTAssertEqual(bundle.stamp.quantization.bits, 4)
            XCTAssertNil(bundle.stamp.defaults.guidance)
            XCTAssertEqual(bundle.limits.dimensionMultiple, 16)
            XCTAssertEqual(bundle.limits.maxSourceImages, 10)
            XCTAssertFalse(bundle.limits.supportsNegativePrompt)
            XCTAssertNil(bundle.limits.fixedSteps)
            let request = ImageGenRequest(prompt: "bakery", negativePrompt: "ignored", width: 1008, height: 1328, steps: 1, guidance: 1, outputDir: root)
            XCTAssertNoThrow(try bundle.limits.validate(request))
            XCTAssertNoThrow(try bundle.limits.validateEdit(width: nil, height: nil, steps: 32, guidance: 1, format: .png, sourceCount: 10, hasMask: false, strength: nil))
            XCTAssertThrowsError(try bundle.limits.validateEdit(width: 1000, height: 1024, steps: 16, guidance: 1, format: .png, sourceCount: 1, hasMask: false, strength: nil))
            XCTAssertThrowsError(try bundle.limits.validateEdit(width: nil, height: nil, steps: 16, guidance: 1, format: .png, sourceCount: 11, hasMask: false, strength: nil))
        }
    }

    func testEveryStampedSideAndVariableStepCount() throws {
        try fixture { root in
            let limits = try Flux3Bundle(modelPath: root).limits
            for side in stride(from: 64, through: 4096, by: 16) {
                XCTAssertNoThrow(try limits.validate(ImageGenRequest(prompt: "x", negativePrompt: "ignored",
                    width: side, height: 1008, steps: 101, guidance: 1, outputDir: root)))
                XCTAssertNoThrow(try limits.validateEdit(width: side, height: 1328, steps: 1, guidance: 1,
                    format: .png, sourceCount: 10, hasMask: false, strength: nil, negativePrompt: "ignored"))
            }
            for steps in [1, 4, 8, 16, 64, 101] {
                let schedule = try Flux3Schedule(steps: steps, alpha: 6.93)
                XCTAssertEqual(schedule.timesteps.count, steps + 1)
                XCTAssertEqual(schedule.timesteps.first, 1)
                XCTAssertEqual(schedule.timesteps.last, 0)
            }
        }
    }

    func testFailClosedBeforeTensors() throws {
        try fixture { root in
            try FileManager.default.removeItem(at: root.appendingPathComponent("vae/0.safetensors"))
            XCTAssertThrowsError(try Flux3Bundle(modelPath: root))
        }
        try fixture { root in
            try Data(#"{"num_hidden_layers":36}"#.utf8).write(to: root.appendingPathComponent("text_encoder/config.json"))
            XCTAssertThrowsError(try Flux3Bundle(modelPath: root))
        }
        for (key, value) in [("hidden_size", 5120), ("intermediate_size", 10000), ("rope_theta", 1000000)] {
            try fixture { root in
                let url = root.appendingPathComponent("text_encoder/config.json")
                var raw = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
                raw[key] = value
                try JSONSerialization.data(withJSONObject: raw).write(to: url)
                XCTAssertThrowsError(try Flux3Bundle(modelPath: root))
            }
        }
        for replacement in ["{}", #"{"osaurus_image":{"family":"flux3-image-klein-base"}}"#] {
            try fixture { root in
                try Data(replacement.utf8).write(to: root.appendingPathComponent("model_index.json"))
                XCTAssertThrowsError(try Flux3Bundle(modelPath: root))
            }
        }
        try fixture { root in
            try Data(#"{"weight_map":{"w":"../outside.safetensors"}}"#.utf8).write(to: root.appendingPathComponent("vae/model.safetensors.index.json"))
            XCTAssertThrowsError(try Flux3Bundle(modelPath: root))
        }
    }

    func testSchedulerMetadataAndVariantFailClosed() throws {
        try fixture { root in
            try Data(#"{"_class_name":"OtherScheduler","num_inference_steps":16,"alpha":6.93,"flow_map":true}"#.utf8)
                .write(to: root.appendingPathComponent("scheduler/scheduler_config.json"))
            XCTAssertThrowsError(try Flux3Bundle(modelPath: root))
        }
        for (key, value) in [("variant", "base"), ("family", "flux2-klein")] {
            try fixture { root in
                let file = root.appendingPathComponent("model_index.json")
                var index = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as! [String: Any]
                var stamp = index["osaurus_image"] as! [String: Any]
                stamp[key] = value; index["osaurus_image"] = stamp
                try JSONSerialization.data(withJSONObject: index).write(to: file)
                XCTAssertThrowsError(try Flux3Bundle(modelPath: root))
            }
        }
    }

    func testNegativePromptPolicyRoundTripAndCompatibility() throws {
        try fixture { root in
            let limits = try Flux3Bundle(modelPath: root).limits
            let roundTrip = try JSONDecoder().decode(FluxModelLimits.self, from: JSONEncoder().encode(limits))
            XCTAssertTrue(roundTrip.ignoresNegativePrompt)
            XCTAssertNoThrow(try roundTrip.validateEdit(width: nil, height: nil, steps: 16, guidance: 1,
                format: .png, sourceCount: 1, hasMask: false, strength: nil, negativePrompt: "must be ignored"))
            let old = Data(#"{"dimensionMultiple":16,"minDimension":16,"minimumSteps":1,"supportsNegativePrompt":false,"supportsEditNegativePrompt":false,"supportsStrength":false,"supportsMask":false,"maxSourceImages":0,"supportedFormats":["png"]}"#.utf8)
            let legacy = try JSONDecoder().decode(FluxModelLimits.self, from: old)
            XCTAssertFalse(legacy.ignoresNegativePrompt)
            XCTAssertThrowsError(try legacy.validate(ImageGenRequest(prompt: "x", negativePrompt: "x", outputDir: root)))
        }
    }

    func testTextEncoderTokenizerIsDiscovered() throws {
        try fixture { root in
            let local = try MLXStudioModelStore.inspect(directory: root)
            XCTAssertTrue(local.components.contains(.tokenizer))
            XCTAssertTrue(local.blockedReasons.isEmpty)
        }
    }

    func testSafetensorsHeaderRejectsTruncationLayoutAndQuantizationMismatch() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("flux3-header-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        func write(_ name: String, tensors: [String: Any], bytes: Int, bits: Int = 4) throws -> URL {
            var object = tensors
            object["__metadata__"] = ["quantization_level": String(bits), "mflux_version": "0.23.0"]
            let header = try JSONSerialization.data(withJSONObject: object)
            var size = UInt64(header.count).littleEndian
            var file = withUnsafeBytes(of: &size) { Data($0) }
            file.append(header); file.append(Data(repeating: 0, count: bytes))
            let url = root.appendingPathComponent(name)
            try file.write(to: url)
            return url
        }
        let valid: [String: Any] = ["a.weight": ["dtype": "BF16", "shape": [2], "data_offsets": [0, 4]]]
        let good = try write("good.safetensors", tensors: valid, bytes: 4)
        XCTAssertEqual(try Flux3WeightManifest.readShard(good, bits: 4)["a.weight"], .init(dtype: "BF16", shape: [2]))
        XCTAssertThrowsError(try Flux3WeightManifest.readShard(good, bits: 6))
        let short = try write("short.safetensors", tensors: valid, bytes: 3)
        XCTAssertThrowsError(try Flux3WeightManifest.readShard(short, bits: 4))
        let trailing = try write("trailing.safetensors", tensors: valid, bytes: 5)
        XCTAssertThrowsError(try Flux3WeightManifest.readShard(trailing, bits: 4))
        let wrongShape = try write("shape.safetensors", tensors: ["a.weight": ["dtype": "BF16", "shape": [3], "data_offsets": [0, 4]]], bytes: 4)
        XCTAssertThrowsError(try Flux3WeightManifest.readShard(wrongShape, bits: 4))
        let overlapping = try write("overlap.safetensors", tensors: [
            "a.weight": ["dtype": "BF16", "shape": [2], "data_offsets": [0, 4]],
            "b.weight": ["dtype": "BF16", "shape": [2], "data_offsets": [2, 6]],
        ], bytes: 6)
        XCTAssertThrowsError(try Flux3WeightManifest.readShard(overlapping, bits: 4))
        let truncated = root.appendingPathComponent("prefix.safetensors")
        try Data([1, 2, 3]).write(to: truncated)
        XCTAssertThrowsError(try Flux3WeightManifest.readShard(truncated, bits: 4))
        let oversized = root.appendingPathComponent("oversized-header.safetensors")
        var length = UInt64(100_000_001).littleEndian
        try withUnsafeBytes(of: &length) { Data($0) }.write(to: oversized)
        let sparse = try FileHandle(forWritingTo: oversized)
        try sparse.truncate(atOffset: 100_000_009) // sparse file, no tensor data written
        try sparse.close()
        XCTAssertThrowsError(try Flux3WeightManifest.readShard(oversized, bits: 4))
    }

    func testLocalBundleMetadataWhenRequested() throws {
        guard let path = ProcessInfo.processInfo.environment["FLUX3_STAGED_BUNDLES"] else {
            throw XCTSkip("Set FLUX3_STAGED_BUNDLES for read-only local stamp preflight")
        }
        for bits in [4, 6, 8] {
            let root = URL(fileURLWithPath: path).appendingPathComponent("FLUX-3-Image-Klein-mflux-\(bits)bit")
            let bundle = try Flux3Bundle(modelPath: root)
            XCTAssertEqual(bundle.stamp.quantization.bits, bits)
            XCTAssertEqual(bundle.stamp.defaults.steps, 16)
            XCTAssertEqual(bundle.scheduler.alpha, 6.93)
            let audit = try Flux3WeightManifest.validate(modelPath: root, bundle: bundle)
            XCTAssertEqual(audit.tensors, ["transformer": 905, "text_encoder": 805, "vae": 266])
            XCTAssertEqual(audit.affineLayers, ["transformer": 273, "text_encoder": 225, "vae": 8])
            print("[Flux3HeaderAudit] bits=\(bits) tensors=\(audit.tensors) affine_layers=\(audit.affineLayers)")
        }
    }

    func testFlowMapScheduleAndEulerReference() throws {
        // mflux d587aac: alpha*t/(1+(alpha-1)*t), float32, N=16.
        let expected: [Float] = [1, 0.9904716611, 0.9798020720, 0.9677731395, 0.9541074038,
            0.9384464025, 0.9203186035, 0.8990918994, 0.8738965988, 0.8435055017,
            0.8061264157, 0.7590360641, 0.6978851557, 0.6152707338, 0.4974874258, 0.3160054684, 0]
        let schedule = try Flux3Schedule(steps: 16, alpha: 6.93)
        // Values evaluated with Python struct float32 rounding after each operation; no MLX import.
        XCTAssertEqual(schedule.timesteps.count, expected.count)
        for (actual, reference) in zip(schedule.timesteps, expected) { XCTAssertEqual(actual, reference, accuracy: 1e-7) }
        XCTAssertTrue(zip(schedule.timesteps, schedule.timesteps.dropFirst()).allSatisfy { $0 > $1 })
        let four = try Flux3Schedule(steps: 4, alpha: 6.93)
        XCTAssertEqual(four.timesteps[1], 0.9541074, accuracy: 1e-6)
        let one = try Flux3Schedule(steps: 1, alpha: 6.93)
        XCTAssertEqual(try one.step(latents: [1, 2, -3], velocity: [2, -4, 0.5], index: 0), [-1, 6, -3.5])
        XCTAssertThrowsError(try Flux3Schedule(steps: 0, alpha: 6.93))
        XCTAssertThrowsError(try Flux3Schedule(steps: 1, alpha: .nan))
    }

    func testPromptPaddingAndPositionContract() {
        let short = Flux3PromptPlan(tokens: [10, 11, 12], padID: 99, maxTokens: 4096, padMultiple: 80)
        XCTAssertEqual(short.ids.count, 80)
        XCTAssertEqual(Array(short.ids.prefix(4)), [10, 11, 12, 99])
        XCTAssertEqual(Array(short.positions.prefix(5)), [0, 1, 2, 1, 1])
        XCTAssertEqual(short.mask.reduce(0, +), 3)
        let capped = Flux3PromptPlan(tokens: Array(0..<4200), padID: 99, maxTokens: 4096, padMultiple: 80)
        XCTAssertEqual(capped.ids.count, 4096)
        XCTAssertEqual(capped.ids.last, 4095)
        XCTAssertEqual(capped.mask.last, 1)
        XCTAssertEqual(Flux3PromptPlan(tokens: Array(0..<80), padID: 99, maxTokens: 4096, padMultiple: 80).ids.count, 80)
    }

    func testFourAxisRotaryAndReferenceIDs() {
        let ids = Flux3PromptPlan.imageIDs(height: 2, width: 2, layer: -2)
        XCTAssertEqual(ids, [[0, 0, 0, -2], [0, 0, 0.5, -2], [0, 0.5, 0, -2], [0, 0.5, 0.5, -2]])
        XCTAssertEqual(Flux3PromptPlan.textIDs(length: 2), [[0, 0, 0, 0], [0, 0, 0, 1]])
        let table = Flux3RotaryTables.tables(ids: [[0, 0.5, 1, -2]], axes: [2, 2, 2, 2], theta: 10000)
        let expectedCos: [Float] = [1, 0.8775825619, 0.5403023059, -0.4161468365]
        let expectedSin: [Float] = [0, 0.4794255386, 0.8414709848, -0.9092974268]
        for (a, b) in zip(table.cos, expectedCos) { XCTAssertEqual(a, b, accuracy: 1e-7) }
        for (a, b) in zip(table.sin, expectedSin) { XCTAssertEqual(a, b, accuracy: 1e-7) }
    }

    func testReferencePixelWorkHonorsCooperativeCancellation() throws {
        let image = try Flux3Pixels(width: 2, height: 2, rgb: Array(repeating: 127, count: 12))
        let interrupt = FluxInterrupt(); interrupt.cancel()
        XCTAssertThrowsError(try FluxInterruptScope.$current.withValue(interrupt) {
            try image.resized(width: 4, height: 4)
        }) { XCTAssertTrue($0 is CancellationError) }
    }

    func testImageReaderMatchesPillowRGBAndEXIF() throws {
        struct Row: Decodable { let file: String; let width, height: Int; let output: [UInt8] }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Resources/Flux3")
        let rows = try JSONDecoder().decode([Row].self, from: Data(contentsOf: root.appendingPathComponent("image-read-parity.json")))
        for row in rows {
            let file = root.appendingPathComponent(row.file)
            let image = try Flux3ImageReader.read(file)
            XCTAssertEqual(image.width, row.width, row.file); XCTAssertEqual(image.height, row.height, row.file)
            XCTAssertEqual(image.rgb, row.output, row.file)
            XCTAssertEqual(try Flux3ImageReader.dimensions(file), .init(row.width, row.height), row.file)
        }
    }

    func testReferencePixelsMatchPillowAndMfluxFixtures() throws {
        struct Row: Decodable {
            let kind: String
            let width, height, outWidth, outHeight: Int
            let input, output: [UInt8]
        }
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Resources/Flux3/pixel-parity.json")
        let rows = try JSONDecoder().decode([Row].self, from: Data(contentsOf: url))
        try fixture { root in
            let limits = try Flux3Bundle(modelPath: root).stamp.limits
            for row in rows {
                let input = try Flux3Pixels(width: row.width, height: row.height, rgb: row.input)
                let output: Flux3Pixels
                switch row.kind {
                case "resize": output = try input.resized(width: row.outWidth, height: row.outHeight)
                case "single": output = try input.prepared(plan: Flux3ReferencePlan.single(.init(row.width, row.height), limits: limits), single: true)
                case "multiple": output = try input.prepared(plan: Flux3ReferencePlan.multiple(.init(row.width, row.height), limits: limits), single: false)
                default: output = try input.oriented(Int(row.kind.dropFirst("orientation".count))!)
                }
                XCTAssertEqual(output.width, row.outWidth, row.kind)
                XCTAssertEqual(output.height, row.outHeight, row.kind)
                XCTAssertEqual(output.rgb, row.output, row.kind)
            }
        }
    }

    func testReferenceCanvasAndCropMatchMflux() throws {
        try fixture { root in
            let limits = try Flux3Bundle(modelPath: root).stamp.limits
            let aligned = try Flux3ReferencePlan.single(.init(1024, 1024), limits: limits)
            XCTAssertEqual(aligned.canvas, .init(1024, 1024)); XCTAssertNil(aligned.crop)
            let padded = try Flux3ReferencePlan.single(.init(1001, 777), limits: limits)
            XCTAssertEqual(padded.canvas, .init(1008, 784))
            XCTAssertEqual(padded.crop?.left, 3); XCTAssertEqual(padded.crop?.top, 3)
            XCTAssertEqual(padded.crop?.size, .init(1001, 777))
            XCTAssertEqual(padded.output(width: nil, height: nil, single: true).crop, padded.crop)
            XCTAssertEqual(padded.output(width: 1001, height: 777, single: true).canvas, padded.canvas)
            XCTAssertNil(padded.output(width: 1024, height: nil, single: true).crop)
            let large = try Flux3ReferencePlan.single(.init(8000, 4000), limits: limits)
            XCTAssertLessThanOrEqual(large.canvas.width * large.canvas.height, limits.singleReferenceMaxPixels)
            XCTAssertLessThanOrEqual(large.canvas.width, 4096)
            let multi = try Flux3ReferencePlan.multiple(.init(2048, 1024), limits: limits)
            XCTAssertEqual(multi.resized, .init(1448, 724))
            XCTAssertEqual(multi.canvas, .init(1440, 720))
            XCTAssertEqual(multi.crop?.left, 4); XCTAssertEqual(multi.crop?.top, 2)
            XCTAssertThrowsError(try Flux3ReferencePlan.multiple(.init(10, 100), limits: limits))
            XCTAssertTrue(try Flux3ReferencePlan.single(.init(1, 1), limits: limits).symmetricPadding)
        }
    }
}
