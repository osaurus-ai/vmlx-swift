import XCTest
import CoreGraphics
import ImageIO
@preconcurrency import MLX
@testable import vMLXFluxKit
@testable import vMLXFluxModels

final class QwenImageEditSupportTests: XCTestCase {

    func testPreprocessPlanMatchesMFluxAspectRatioSizing() throws {
        let plan = try QwenImageEditPreprocessPlan(
            sourceWidth: 1536,
            sourceHeight: 1024,
            requestedWidth: nil,
            requestedHeight: nil,
            steps: 20,
            guidance: 4.0)

        XCTAssertEqual(plan.outputWidth, 1248)
        XCTAssertEqual(plan.outputHeight, 832)
        XCTAssertEqual(plan.vlWidth, 480)
        XCTAssertEqual(plan.vlHeight, 320)
        XCTAssertEqual(plan.vaeWidth, 1248)
        XCTAssertEqual(plan.vaeHeight, 832)
        XCTAssertEqual(plan.conditioningPatchRows, 52)
        XCTAssertEqual(plan.conditioningPatchColumns, 78)
    }

    func testPreprocessPlanFloorsExplicitOutputDimensionsToVAEGrid() throws {
        let plan = try QwenImageEditPreprocessPlan(
            sourceWidth: 480,
            sourceHeight: 640,
            requestedWidth: 513,
            requestedHeight: 1025,
            steps: 4,
            guidance: 3.0)

        XCTAssertEqual(plan.outputWidth, 512)
        XCTAssertEqual(plan.outputHeight, 1024)
        XCTAssertEqual(plan.vlWidth, 320)
        XCTAssertEqual(plan.vlHeight, 448)
        XCTAssertEqual(plan.vaeWidth, 896)
        XCTAssertEqual(plan.vaeHeight, 1184)
    }

    func testImageIdsUseOneInFirstAxisAndRowColumnPatchCoordinates() throws {
        let ids = try QwenImageEditPreprocessPlan.imageIDs(height: 32, width: 48)

        XCTAssertEqual(ids.count, 6)
        XCTAssertEqual(ids[0], [1, 0, 0])
        XCTAssertEqual(ids[1], [1, 0, 1])
        XCTAssertEqual(ids[2], [1, 0, 2])
        XCTAssertEqual(ids[3], [1, 1, 0])
        XCTAssertEqual(ids[4], [1, 1, 1])
        XCTAssertEqual(ids[5], [1, 1, 2])
    }

    func testPreprocessPlanRejectsInvalidDimensions() {
        XCTAssertThrowsError(try QwenImageEditPreprocessPlan(
            sourceWidth: 0,
            sourceHeight: 1024,
            requestedWidth: nil,
            requestedHeight: nil,
            steps: 20,
            guidance: 4.0))

        XCTAssertThrowsError(try QwenImageEditPreprocessPlan.imageIDs(height: 31, width: 48))
    }

    func testPreprocessPlanReadsSourceImageDimensions() throws {
        let source = try makePNG(width: 1536, height: 1024)

        let plan = try QwenImageEditPreprocessPlan(
            sourceImage: source,
            requestedWidth: nil,
            requestedHeight: nil,
            steps: 20,
            guidance: 4.0)

        XCTAssertEqual(plan.outputWidth, 1248)
        XCTAssertEqual(plan.outputHeight, 832)
        XCTAssertEqual(plan.vlWidth, 480)
        XCTAssertEqual(plan.vlHeight, 320)
    }

    func testVisionInputMatchesQwenVLProcessorShapeAndNormalization() throws {
        let source = try makePNG(width: 512, height: 512, rgba: (255, 128, 0, 255))
        let plan = try QwenImageEditPreprocessPlan(
            sourceImage: source,
            requestedWidth: nil,
            requestedHeight: nil,
            steps: 20,
            guidance: 4.0)

        let input = try QwenImageEditPreprocessor.visionInput(
            sourceImage: source,
            plan: plan)

        XCTAssertEqual(input.resizedWidth, 392)
        XCTAssertEqual(input.resizedHeight, 392)
        XCTAssertEqual(input.imageGridTHW, [1, 28, 28])
        XCTAssertEqual(input.imageTokenCount, 196)
        XCTAssertEqual(input.pixelValues.shape, [784, 1176])

        let values = input.pixelValues.asArray(Float.self)
        XCTAssertEqual(values.count, 784 * 1176)
        let red = Float((1.0 - 0.48145466) / 0.26862954)
        let green = Float(((128.0 / 255.0) - 0.4578275) / 0.26130258)
        let blue = Float((0.0 - 0.40821073) / 0.27577711)
        XCTAssertEqual(values[0], red, accuracy: 0.001)
        XCTAssertEqual(values[14 * 14 * 2], green, accuracy: 0.001)
        XCTAssertEqual(values[14 * 14 * 4], blue, accuracy: 0.001)
    }

    func testVisionLanguagePromptExpandsImagePadTokensAndKeepsEditDropIndex() throws {
        let prompt = try QwenImageEditPreprocessor.visionLanguagePrompt(
            prompt: "make the background blue",
            imageTokenCounts: [196])

        XCTAssertEqual(prompt.templateDropIndex, 64)
        XCTAssertEqual(prompt.imageTokenCounts, [196])
        XCTAssertTrue(prompt.formattedText.contains("<|im_start|>system\nDescribe the key features of the input image"))
        XCTAssertTrue(prompt.formattedText.contains("<|im_start|>user\nPicture 1: <|vision_start|>"))
        XCTAssertTrue(prompt.formattedText.contains("<|vision_end|>make the background blue<|im_end|>"))
        XCTAssertEqual(
            prompt.formattedText.components(separatedBy: "<|image_pad|>").count - 1,
            196)
    }

    func testVisionFeaturesMatchMergedImageTokenCountAndHiddenSize() throws {
        let features = QwenImageEditVisionFeatures(
            imageFeatures: MLXArray.zeros([196, 3584], dtype: .float32),
            imageGridTHW: [1, 28, 28])

        XCTAssertEqual(features.imageTokenCount, 196)
        XCTAssertEqual(features.hiddenSize, 3584)
        XCTAssertNoThrow(try features.validateMatches(promptImageTokenCount: 196))
        XCTAssertThrowsError(try features.validateMatches(promptImageTokenCount: 195))
        XCTAssertThrowsError(try QwenImageEditVisionFeatures(
            imageFeatures: MLXArray.zeros([196, 3583], dtype: .float32),
            imageGridTHW: [1, 28, 28]).validateMatches(promptImageTokenCount: 196))
    }

    func testPromptEmbeddingsKeepPostTemplateMaskAndHiddenSize() throws {
        let embeddings = QwenImageEditPromptEmbeddings(
            promptEmbeds: MLXArray.zeros([1, 212, 3584], dtype: .float32),
            attentionMask: MLXArray.ones([1, 212], dtype: .int32),
            templateDropIndex: 64,
            sourceSequenceLength: 276)

        XCTAssertEqual(embeddings.sequenceLength, 212)
        XCTAssertEqual(embeddings.hiddenSize, 3584)
        XCTAssertNoThrow(try embeddings.validate())
        XCTAssertThrowsError(try QwenImageEditPromptEmbeddings(
            promptEmbeds: MLXArray.zeros([1, 212, 3583], dtype: .float32),
            attentionMask: MLXArray.ones([1, 212], dtype: .int32),
            templateDropIndex: 64,
            sourceSequenceLength: 276).validate())
    }

    func testTransformerInputsConcatenateTargetAndConditioningLatents() throws {
        let target = MLXArray.zeros([1, 256, 64], dtype: .float32)
        let conditioning = QwenImageEditConditioningLatents(
            latents: MLXArray.ones([1, 4096, 64], dtype: .float32),
            imageIDs: MLXArray.zeros([1, 4096, 3], dtype: .float32),
            patchRows: 64,
            patchColumns: 64)

        let inputs = try QwenImageEditTransformerInputs(
            targetLatents: target,
            conditioning: conditioning)

        XCTAssertEqual(inputs.hiddenStates.shape, [1, 4352, 64])
        XCTAssertEqual(inputs.targetLatentCount, 256)
        XCTAssertEqual(inputs.conditioningLatentCount, 4096)
        XCTAssertEqual(inputs.imageShapes.map { [$0.frame, $0.height, $0.width] }, [[1, 16, 16], [1, 64, 64]])
        XCTAssertEqual(inputs.targetVelocitySlice.shape, [1, 256, 64])
    }

    func testTransformerInputsAcceptNonSquareTargetGridFromPlan() throws {
        let target = MLXArray.zeros([1, 4056, 64], dtype: .float32)
        let conditioning = QwenImageEditConditioningLatents(
            latents: MLXArray.ones([1, 4056, 64], dtype: .float32),
            imageIDs: MLXArray.zeros([1, 4056, 3], dtype: .float32),
            patchRows: 52,
            patchColumns: 78)

        let inputs = try QwenImageEditTransformerInputs(
            targetLatents: target,
            targetPatchRows: 52,
            targetPatchColumns: 78,
            conditioning: conditioning)

        XCTAssertEqual(inputs.hiddenStates.shape, [1, 8112, 64])
        XCTAssertEqual(inputs.targetLatentCount, 4056)
        XCTAssertEqual(inputs.conditioningLatentCount, 4056)
        XCTAssertEqual(inputs.imageShapes.map { [$0.frame, $0.height, $0.width] }, [[1, 52, 78], [1, 52, 78]])
        XCTAssertEqual(inputs.targetVelocitySlice.shape, [1, 4056, 64])
    }

    func testQwenEditRoPECombinesTargetAndConditioningImageGrids() {
        let ((imgCos, imgSin), (txtCos, txtSin)) = QwenRoPE.freqs(
            imageShapes: [
                (frame: 1, height: 16, width: 16),
                (frame: 1, height: 64, width: 64),
            ],
            txtLen: 212,
            dtype: .float32)

        XCTAssertEqual(imgCos.shape, [4352, 64])
        XCTAssertEqual(imgSin.shape, [4352, 64])
        XCTAssertEqual(txtCos.shape, [212, 64])
        XCTAssertEqual(txtSin.shape, [212, 64])
    }

    func testEditDenoiseResultExposesOnlyTargetVelocitySlice() throws {
        let result = try QwenImageEditDenoiseResult(
            combinedVelocity: MLXArray.zeros([1, 4352, 64], dtype: .float32),
            targetLatentCount: 256,
            imageShapes: [
                QwenImageEditImageShape(frame: 1, height: 16, width: 16),
                QwenImageEditImageShape(frame: 1, height: 64, width: 64),
            ])

        XCTAssertEqual(result.combinedVelocity.shape, [1, 4352, 64])
        XCTAssertEqual(result.targetVelocity.shape, [1, 256, 64])
        XCTAssertEqual(result.targetLatentCount, 256)
        XCTAssertEqual(result.conditioningLatentCount, 4096)
    }

    func testQwenGuidanceRescalesGuidedNoiseToConditionalNorm() {
        let positive = MLXArray([3, 4, 0, 0], [1, 2, 2]).asType(.float32)
        let negative = MLXArray([0, 0, 1, 0], [1, 2, 2]).asType(.float32)

        let guided = QwenGuidance.computeGuidedNoise(
            positive: positive,
            negative: negative,
            guidance: 4)
        let norms = sqrt(sum(guided * guided, axis: -1))
            .asArray(Float.self)

        XCTAssertEqual(norms[0], 5, accuracy: 0.001)
        XCTAssertEqual(norms[1], 0, accuracy: 0.0001)
    }

    func testVAEInputUsesMinusOneToOneNCHWAtConditioningSize() throws {
        let source = try makePNG(width: 512, height: 512, rgba: (255, 128, 0, 255))
        let plan = try QwenImageEditPreprocessPlan(
            sourceImage: source,
            requestedWidth: nil,
            requestedHeight: nil,
            steps: 20,
            guidance: 4.0)

        let input = try QwenImageEditPreprocessor.vaeInput(
            sourceImage: source,
            plan: plan)

        XCTAssertEqual(input.tensor.shape, [1, 3, 1024, 1024])
        let values = input.tensor.asArray(Float.self)
        XCTAssertEqual(values[0], 1.0, accuracy: 0.001)
        XCTAssertEqual(values[1024 * 1024], (128.0 / 255.0) * 2.0 - 1.0, accuracy: 0.001)
        XCTAssertEqual(values[1024 * 1024 * 2], -1.0, accuracy: 0.001)
    }

    func testConditioningLatentsPackEncodedVAELatentsWithImageIDs() throws {
        let encoded = MLXArray((0 ..< (16 * 4 * 6)).map(Float.init), [1, 16, 4, 6])
            .asType(.float32)

        let conditioning = try QwenImageEditPreprocessor.conditioningLatents(
            encodedLatents: encoded,
            height: 32,
            width: 48)

        XCTAssertEqual(conditioning.latents.shape, [1, 6, 64])
        XCTAssertEqual(conditioning.imageIDs.shape, [1, 6, 3])
        XCTAssertEqual(conditioning.patchRows, 2)
        XCTAssertEqual(conditioning.patchColumns, 3)

        let packed = conditioning.latents.asArray(Float.self)
        XCTAssertEqual(Array(packed[0 ..< 8]), [0, 1, 6, 7, 24, 25, 30, 31])
        XCTAssertEqual(Array(packed[64 ..< 72]), [2, 3, 8, 9, 26, 27, 32, 33])
        XCTAssertEqual(Array(packed[192 ..< 200]), [12, 13, 18, 19, 36, 37, 42, 43])

        let ids = conditioning.imageIDs.asArray(Float.self)
        XCTAssertEqual(Array(ids[0 ..< 9]), [1, 0, 0, 1, 0, 1, 1, 0, 2])
    }

    func testQwenImageEditReportsLoadFailureAsFailedEvent() async throws {
        let model = try makeTemporaryQwenImageEditBundle()
        let source = try makePNG(width: 1536, height: 1024)
        let outputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-edit-output-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: outputDir)
        }

        let editor = try QwenImageEdit(modelPath: model, quantize: 4)
        let request = ImageEditRequest(
            prompt: "make the background blue",
            sourceImage: source,
            width: nil,
            height: nil,
            steps: 20,
            guidance: 4.0,
            outputDir: outputDir)

        var failedMessage: String?
        do {
            for try await event in editor.edit(request) {
                if case .failed(let message, _) = event {
                    failedMessage = message
                }
            }
        } catch {
            XCTFail("edit stream should report load/runtime errors as failed events, got throw: \(error)")
        }
        let message = try XCTUnwrap(failedMessage)
        XCTAssertFalse(message.contains("Qwen2.5-VL vision encoder"))
        XCTAssertFalse(message.contains("notImplemented"))
    }

    private func makePNG(
        width: Int,
        height: Int,
        rgba: (UInt8, UInt8, UInt8, UInt8) = (255, 255, 255, 255)
    ) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-edit-source-\(UUID().uuidString).png")
        addTeardownBlock {
            try? FileManager.default.removeItem(at: url)
        }
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: height * bytesPerRow)
        for offset in stride(from: 0, to: pixels.count, by: 4) {
            pixels[offset] = rgba.0
            pixels[offset + 1] = rgba.1
            pixels[offset + 2] = rgba.2
            pixels[offset + 3] = rgba.3
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let image = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent),
              let destination = CGImageDestinationCreateWithURL(
                url as CFURL,
                "public.png" as CFString,
                1,
                nil)
        else {
            throw NSError(domain: "QwenImageEditSupportTests", code: 1)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw NSError(domain: "QwenImageEditSupportTests", code: 2)
        }
        return url
    }

    private func makeTemporaryQwenImageEditBundle() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-edit-bundle-\(UUID().uuidString)", isDirectory: true)
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
            keys: [
                "encoder.embed_tokens.weight",
                "encoder.layers.0.self_attn.q_proj.weight",
                "encoder.norm.weight",
                "encoder.visual.patch_embed.proj.weight",
                "encoder.visual.blocks.0.attn.qkv.weight",
                "encoder.visual.blocks.31.attn.qkv.weight",
                "encoder.visual.merger.mlp_1.weight",
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
