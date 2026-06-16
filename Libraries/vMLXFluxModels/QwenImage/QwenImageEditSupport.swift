import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom
import VMLXTokenizers
import vMLXFluxKit

public struct QwenImageEditPreprocessPlan: Sendable {
    public let outputWidth: Int
    public let outputHeight: Int
    public let vlWidth: Int
    public let vlHeight: Int
    public let vaeWidth: Int
    public let vaeHeight: Int
    public let conditioningPatchRows: Int
    public let conditioningPatchColumns: Int
    public let steps: Int
    public let guidance: Float

    public init(
        sourceImage: URL,
        requestedWidth: Int?,
        requestedHeight: Int?,
        steps: Int,
        guidance: Float
    ) throws {
        let dimensions = try ImageIO.dimensions(of: sourceImage)
        try self.init(
            sourceWidth: dimensions.width,
            sourceHeight: dimensions.height,
            requestedWidth: requestedWidth,
            requestedHeight: requestedHeight,
            steps: steps,
            guidance: guidance)
    }

    public init(
        sourceWidth: Int,
        sourceHeight: Int,
        requestedWidth: Int?,
        requestedHeight: Int?,
        steps: Int,
        guidance: Float
    ) throws {
        guard sourceWidth > 0, sourceHeight > 0 else {
            throw FluxError.invalidRequest("Qwen edit source image dimensions must be positive")
        }
        guard steps > 0 else {
            throw FluxError.invalidRequest("Qwen edit steps must be greater than zero")
        }
        guard guidance.isFinite else {
            throw FluxError.invalidRequest("Qwen edit guidance must be finite")
        }

        let ratio = Double(sourceWidth) / Double(sourceHeight)
        let generated = Self.roundedAreaDimensions(area: 1024 * 1024, ratio: ratio)
        let outputWidth = Self.floorToMultiple(requestedWidth ?? generated.width, multiple: 16)
        let outputHeight = Self.floorToMultiple(requestedHeight ?? generated.height, multiple: 16)
        guard outputWidth > 0, outputHeight > 0 else {
            throw FluxError.invalidRequest("Qwen edit output dimensions must be at least 16 pixels")
        }

        let vl = Self.roundedAreaDimensions(area: 384 * 384, ratio: ratio)
        let vae = Self.roundedAreaDimensions(area: 1024 * 1024, ratio: ratio)

        self.outputWidth = outputWidth
        self.outputHeight = outputHeight
        self.vlWidth = vl.width
        self.vlHeight = vl.height
        self.vaeWidth = vae.width
        self.vaeHeight = vae.height
        self.conditioningPatchRows = vae.height / 16
        self.conditioningPatchColumns = vae.width / 16
        self.steps = steps
        self.guidance = guidance
    }

    public static func imageIDs(height: Int, width: Int) throws -> [[Int]] {
        guard height > 0, width > 0, height % 16 == 0, width % 16 == 0 else {
            throw FluxError.invalidRequest("Qwen edit conditioning image dimensions must be positive multiples of 16")
        }
        let latentHeight = height / 16
        let latentWidth = width / 16
        var ids: [[Int]] = []
        ids.reserveCapacity(latentHeight * latentWidth)
        for row in 0 ..< latentHeight {
            for column in 0 ..< latentWidth {
                ids.append([1, row, column])
            }
        }
        return ids
    }

    private static func roundedAreaDimensions(area: Int, ratio: Double) -> (width: Int, height: Int) {
        let width = sqrt(Double(area) * ratio)
        let height = width / ratio
        return (
            roundToNearestEvenMultiple(width, multiple: 32),
            roundToNearestEvenMultiple(height, multiple: 32)
        )
    }

    private static func roundToNearestEvenMultiple(_ value: Double, multiple: Int) -> Int {
        Int((value / Double(multiple)).rounded(.toNearestOrEven)) * multiple
    }

    private static func floorToMultiple(_ value: Int, multiple: Int) -> Int {
        value / multiple * multiple
    }
}

public struct QwenImageEditVisionInput {
    public let pixelValues: MLXArray
    public let imageGridTHW: [Int]
    public let resizedWidth: Int
    public let resizedHeight: Int

    public var imageTokenCount: Int {
        imageGridTHW.reduce(1, *) / (QwenImageEditPreprocessor.mergeSize * QwenImageEditPreprocessor.mergeSize)
    }
}

public struct QwenImageEditPromptInput {
    public let formattedText: String
    public let imageTokenCounts: [Int]
    public let templateDropIndex: Int
}

public struct QwenImageEditPromptTokens {
    public let inputIDs: MLXArray
    public let attentionMask: MLXArray
    public let sequenceLength: Int
    public let imageTokenCount: Int
    public let templateDropIndex: Int
}

public struct QwenImageEditVisionFeatures {
    public let imageFeatures: MLXArray
    public let imageGridTHW: [Int]

    public var imageTokenCount: Int { imageFeatures.dim(0) }
    public var hiddenSize: Int { imageFeatures.dim(1) }

    public init(imageFeatures: MLXArray, imageGridTHW: [Int]) {
        self.imageFeatures = imageFeatures
        self.imageGridTHW = imageGridTHW
    }

    public func validateMatches(promptImageTokenCount: Int) throws {
        guard imageGridTHW.count == 3 else {
            throw FluxError.invalidRequest("Qwen edit vision grid must be [t,h,w]")
        }
        let expected = try QwenImageEditVisionTransformer.mergedTokenCount(imageGridTHW: imageGridTHW)
        guard imageTokenCount == expected else {
            throw FluxError.invalidRequest(
                "Qwen edit vision feature count \(imageTokenCount) does not match grid-derived count \(expected)")
        }
        guard imageTokenCount == promptImageTokenCount else {
            throw FluxError.invalidRequest(
                "Qwen edit vision feature count \(imageTokenCount) does not match prompt image tokens \(promptImageTokenCount)")
        }
        guard hiddenSize == QwenTextEncoder.hidden else {
            throw FluxError.invalidRequest(
                "Qwen edit vision hidden size \(hiddenSize) does not match Qwen text hidden size \(QwenTextEncoder.hidden)")
        }
    }
}

public struct QwenImageEditPromptEmbeddings {
    public let promptEmbeds: MLXArray
    public let attentionMask: MLXArray
    public let templateDropIndex: Int
    public let sourceSequenceLength: Int

    public var sequenceLength: Int { promptEmbeds.dim(1) }
    public var hiddenSize: Int { promptEmbeds.dim(2) }

    public init(
        promptEmbeds: MLXArray,
        attentionMask: MLXArray,
        templateDropIndex: Int,
        sourceSequenceLength: Int
    ) {
        self.promptEmbeds = promptEmbeds
        self.attentionMask = attentionMask
        self.templateDropIndex = templateDropIndex
        self.sourceSequenceLength = sourceSequenceLength
    }

    public func validate() throws {
        guard promptEmbeds.ndim == 3, promptEmbeds.dim(0) == 1 else {
            throw FluxError.invalidRequest("Qwen edit prompt embeds must have shape [1,seq,hidden]")
        }
        guard attentionMask.shape == [1, sequenceLength] else {
            throw FluxError.invalidRequest(
                "Qwen edit prompt mask shape \(attentionMask.shape) does not match embeds sequence \(sequenceLength)")
        }
        guard sourceSequenceLength >= templateDropIndex,
              sequenceLength == sourceSequenceLength - templateDropIndex
        else {
            throw FluxError.invalidRequest(
                "Qwen edit prompt sequence \(sequenceLength) does not match source \(sourceSequenceLength) minus template drop \(templateDropIndex)")
        }
        guard hiddenSize == QwenTextEncoder.hidden else {
            throw FluxError.invalidRequest(
                "Qwen edit prompt hidden size \(hiddenSize) does not match Qwen text hidden size \(QwenTextEncoder.hidden)")
        }
    }
}

public struct QwenImageEditVAEInput {
    public let tensor: MLXArray
}

public struct QwenImageEditConditioningLatents {
    public let latents: MLXArray
    public let imageIDs: MLXArray
    public let patchRows: Int
    public let patchColumns: Int
}

public struct QwenImageEditImageShape {
    public let frame: Int
    public let height: Int
    public let width: Int
}

public struct QwenImageEditTransformerInputs {
    public let hiddenStates: MLXArray
    public let targetLatentCount: Int
    public let conditioningLatentCount: Int
    public let imageShapes: [QwenImageEditImageShape]

    public var targetVelocitySlice: MLXArray {
        hiddenStates[0..., 0 ..< targetLatentCount, 0...]
    }

    public init(
        targetLatents: MLXArray,
        conditioning: QwenImageEditConditioningLatents
    ) throws {
        guard targetLatents.shape.count == 3 else {
            throw FluxError.invalidRequest("Qwen edit target latents must have shape 1xNx64")
        }
        let targetCount = targetLatents.dim(1)
        let targetSide = Int(Double(targetCount).squareRoot())
        guard targetSide * targetSide == targetCount else {
            throw FluxError.invalidRequest("Qwen edit target latent count must form a square grid for this proof path")
        }
        try self.init(
            targetLatents: targetLatents,
            targetPatchRows: targetSide,
            targetPatchColumns: targetSide,
            conditioning: conditioning)
    }

    public init(
        targetLatents: MLXArray,
        targetPatchRows: Int,
        targetPatchColumns: Int,
        conditioning: QwenImageEditConditioningLatents
    ) throws {
        guard targetLatents.shape.count == 3,
              targetLatents.dim(0) == 1,
              targetLatents.dim(2) == 64
        else {
            throw FluxError.invalidRequest("Qwen edit target latents must have shape 1xNx64")
        }
        guard conditioning.latents.shape.count == 3,
              conditioning.latents.dim(0) == 1,
              conditioning.latents.dim(2) == 64
        else {
            throw FluxError.invalidRequest("Qwen edit conditioning latents must have shape 1xNx64")
        }
        guard conditioning.patchRows > 0, conditioning.patchColumns > 0 else {
            throw FluxError.invalidRequest("Qwen edit conditioning patch grid must be positive")
        }
        guard conditioning.latents.dim(1) == conditioning.patchRows * conditioning.patchColumns else {
            throw FluxError.invalidRequest("Qwen edit conditioning latent count does not match patch grid")
        }
        guard conditioning.imageIDs.shape == [1, conditioning.latents.dim(1), 3] else {
            throw FluxError.invalidRequest("Qwen edit conditioning image IDs must have shape 1xNx3")
        }

        let targetCount = targetLatents.dim(1)
        guard targetPatchRows > 0, targetPatchColumns > 0 else {
            throw FluxError.invalidRequest("Qwen edit target patch grid must be positive")
        }
        guard targetCount == targetPatchRows * targetPatchColumns else {
            throw FluxError.invalidRequest("Qwen edit target latent count does not match target patch grid")
        }

        self.hiddenStates = concatenated([targetLatents, conditioning.latents], axis: 1)
        self.targetLatentCount = targetCount
        self.conditioningLatentCount = conditioning.latents.dim(1)
        self.imageShapes = [
            QwenImageEditImageShape(frame: 1, height: targetPatchRows, width: targetPatchColumns),
            QwenImageEditImageShape(frame: 1, height: conditioning.patchRows, width: conditioning.patchColumns),
        ]
    }
}

public struct QwenImageEditDenoiseResult {
    public let combinedVelocity: MLXArray
    public let targetVelocity: MLXArray
    public let targetLatentCount: Int
    public let conditioningLatentCount: Int
    public let imageShapes: [QwenImageEditImageShape]

    public init(
        combinedVelocity: MLXArray,
        targetLatentCount: Int,
        imageShapes: [QwenImageEditImageShape]
    ) throws {
        guard combinedVelocity.shape.count == 3,
              combinedVelocity.dim(0) == 1,
              combinedVelocity.dim(2) == 64
        else {
            throw FluxError.invalidRequest("Qwen edit combined velocity must have shape 1xNx64")
        }
        guard targetLatentCount > 0, targetLatentCount <= combinedVelocity.dim(1) else {
            throw FluxError.invalidRequest("Qwen edit target velocity slice is outside combined velocity")
        }
        self.combinedVelocity = combinedVelocity
        self.targetVelocity = combinedVelocity[0..., 0 ..< targetLatentCount, 0...]
        self.targetLatentCount = targetLatentCount
        self.conditioningLatentCount = combinedVelocity.dim(1) - targetLatentCount
        self.imageShapes = imageShapes
    }
}

public enum QwenImageEditPreprocessor {
    public static let patchSize = 14
    public static let temporalPatchSize = 2
    public static let mergeSize = 2
    public static let editTemplateStartIndex = 64
    public static let imagePadToken = "<|image_pad|>"
    public static let imageTokenID: Int32 = 151655

    public static func visionLanguagePrompt(
        prompt: String,
        imageTokenCounts: [Int]
    ) throws -> QwenImageEditPromptInput {
        guard !imageTokenCounts.isEmpty else {
            throw FluxError.invalidRequest("Qwen edit prompt requires at least one source image")
        }
        for count in imageTokenCounts where count <= 0 {
            throw FluxError.invalidRequest("Qwen edit image token counts must be positive")
        }

        let imagePrompts = imageTokenCounts.enumerated().map { index, count in
            "Picture \(index + 1): <|vision_start|>"
                + String(repeating: imagePadToken, count: count)
                + "<|vision_end|>"
        }.joined()
        let formatted = "<|im_start|>system\n"
            + "Describe the key features of the input image (color, shape, size, texture, objects, background), "
            + "then explain how the user's text instruction should alter or modify the image. "
            + "Generate a new image that meets the user's requirements while maintaining consistency "
            + "with the original input where appropriate.<|im_end|>\n"
            + "<|im_start|>user\n"
            + imagePrompts
            + prompt
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
        return QwenImageEditPromptInput(
            formattedText: formatted,
            imageTokenCounts: imageTokenCounts,
            templateDropIndex: editTemplateStartIndex)
    }

    public static func visionInput(
        sourceImage: URL,
        plan: QwenImageEditPreprocessPlan
    ) throws -> QwenImageEditVisionInput {
        let resized = try smartResize(height: plan.vlHeight, width: plan.vlWidth)
        let chw = try ImageIO.readRGBValues(
            sourceImage,
            width: resized.width,
            height: resized.height,
            normalization: .openAIClip)
        let gridT = 1
        let gridH = resized.height / patchSize
        let gridW = resized.width / patchSize
        guard resized.height % (patchSize * mergeSize) == 0,
              resized.width % (patchSize * mergeSize) == 0
        else {
            throw FluxError.invalidRequest("Qwen edit VL dimensions must be multiples of \(patchSize * mergeSize)")
        }

        let values = visionPatchValues(
            chw: chw,
            height: resized.height,
            width: resized.width,
            gridH: gridH,
            gridW: gridW)
        let pixelValues = MLXArray(
            values,
            [gridT * gridH * gridW, 3 * temporalPatchSize * patchSize * patchSize]
        ).asType(.float32)
        return QwenImageEditVisionInput(
            pixelValues: pixelValues,
            imageGridTHW: [gridT, gridH, gridW],
            resizedWidth: resized.width,
            resizedHeight: resized.height)
    }

    public static func vaeInput(
        sourceImage: URL,
        plan: QwenImageEditPreprocessPlan
    ) throws -> QwenImageEditVAEInput {
        let tensor = try ImageIO.readRGBTensor(
            sourceImage,
            width: plan.vaeWidth,
            height: plan.vaeHeight,
            normalization: .minusOneToOne)
        return QwenImageEditVAEInput(tensor: tensor)
    }

    public static func conditioningLatents(
        encodedLatents: MLXArray,
        height: Int,
        width: Int
    ) throws -> QwenImageEditConditioningLatents {
        guard height > 0, width > 0, height % 16 == 0, width % 16 == 0 else {
            throw FluxError.invalidRequest("Qwen edit conditioning latent dimensions must be positive multiples of 16")
        }
        guard encodedLatents.dim(0) == 1,
              encodedLatents.dim(1) == 16,
              encodedLatents.dim(2) == height / 8,
              encodedLatents.dim(3) == width / 8
        else {
            throw FluxError.invalidRequest(
                "Qwen edit encoded VAE latents must have shape 1x16x\(height / 8)x\(width / 8)")
        }

        let packed = patchify(encodedLatents, patchSize: 2, inChannels: 16)
        let ids = try QwenImageEditPreprocessPlan.imageIDs(height: height, width: width)
        let idValues = ids.flatMap { $0.map(Float.init) }
        let imageIDs = MLXArray(idValues, [1, ids.count, 3]).asType(.float32)
        return QwenImageEditConditioningLatents(
            latents: packed,
            imageIDs: imageIDs,
            patchRows: height / 16,
            patchColumns: width / 16)
    }

    public static func smartResize(
        height: Int,
        width: Int,
        factor: Int = patchSize * mergeSize,
        minPixels: Int = 56 * 56,
        maxPixels: Int = 28 * 28 * 1280
    ) throws -> (height: Int, width: Int) {
        guard height > 0, width > 0 else {
            throw FluxError.invalidRequest("Qwen edit image dimensions must be positive")
        }
        let aspect = Double(max(height, width)) / Double(min(height, width))
        guard aspect <= 200 else {
            throw FluxError.invalidRequest("Qwen edit image aspect ratio must be <= 200")
        }

        var resizedHeight = Int((Double(height) / Double(factor)).rounded(.toNearestOrEven)) * factor
        var resizedWidth = Int((Double(width) / Double(factor)).rounded(.toNearestOrEven)) * factor
        if resizedHeight * resizedWidth > maxPixels {
            let beta = sqrt(Double(height * width) / Double(maxPixels))
            resizedHeight = max(factor, Int(floor(Double(height) / beta / Double(factor))) * factor)
            resizedWidth = max(factor, Int(floor(Double(width) / beta / Double(factor))) * factor)
        } else if resizedHeight * resizedWidth < minPixels {
            let beta = sqrt(Double(minPixels) / Double(height * width))
            resizedHeight = Int(ceil(Double(height) * beta / Double(factor))) * factor
            resizedWidth = Int(ceil(Double(width) * beta / Double(factor))) * factor
        }
        return (resizedHeight, resizedWidth)
    }

    private static func visionPatchValues(
        chw: [Float],
        height: Int,
        width: Int,
        gridH: Int,
        gridW: Int
    ) -> [Float] {
        let plane = height * width
        let patchVectorLength = 3 * temporalPatchSize * patchSize * patchSize
        var values: [Float] = []
        values.reserveCapacity(gridH * gridW * patchVectorLength)

        for blockH in 0 ..< (gridH / mergeSize) {
            for blockW in 0 ..< (gridW / mergeSize) {
                for mergeH in 0 ..< mergeSize {
                    for mergeW in 0 ..< mergeSize {
                        let patchY = (blockH * mergeSize + mergeH) * patchSize
                        let patchX = (blockW * mergeSize + mergeW) * patchSize
                        for channel in 0 ..< 3 {
                            for _ in 0 ..< temporalPatchSize {
                                for y in 0 ..< patchSize {
                                    let rowOffset = (patchY + y) * width + patchX
                                    let channelOffset = channel * plane
                                    for x in 0 ..< patchSize {
                                        values.append(chw[channelOffset + rowOffset + x])
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return values
    }
}

public final class QwenImageEditPromptTokenizer {
    private let tok: any VMLXTokenizers.Tokenizer

    public init(modelPath: URL) async throws {
        tok = try await AutoTokenizer.from(
            modelFolder: modelPath.appendingPathComponent("tokenizer"),
            strict: false)
    }

    public func tokenize(_ prompt: QwenImageEditPromptInput) -> QwenImageEditPromptTokens {
        let ids = tok.encode(text: prompt.formattedText, addSpecialTokens: false)
        let inputIDs = MLXArray(ids.map(Int32.init)).reshaped([1, ids.count])
        let attentionMask = MLXArray([Int32](repeating: 1, count: ids.count)).reshaped([1, ids.count])
        let imageTokens = ids.reduce(0) { count, id in
            count + (Int32(id) == QwenImageEditPreprocessor.imageTokenID ? 1 : 0)
        }
        return QwenImageEditPromptTokens(
            inputIDs: inputIDs,
            attentionMask: attentionMask,
            sequenceLength: ids.count,
            imageTokenCount: imageTokens,
            templateDropIndex: prompt.templateDropIndex)
    }
}

public enum QwenImageEditConditioner {
    public static func encode(
        modelPath: URL,
        sourceImage: URL,
        plan: QwenImageEditPreprocessPlan
    ) throws -> QwenImageEditConditioningLatents {
        let store = MFluxStore(try WeightLoader.load(from: modelPath))
        return try encode(store: store, sourceImage: sourceImage, plan: plan)
    }

    static func encode(
        store: MFluxStore,
        sourceImage: URL,
        plan: QwenImageEditPreprocessPlan
    ) throws -> QwenImageEditConditioningLatents {
        let encoder = try Qwen3DVAEEncoder(store: store)
        let vaeInput = try QwenImageEditPreprocessor.vaeInput(sourceImage: sourceImage, plan: plan)
        let encoded = encoder.encode(vaeInput.tensor)
        eval(encoded)
        return try QwenImageEditPreprocessor.conditioningLatents(
            encodedLatents: encoded,
            height: plan.vaeHeight,
            width: plan.vaeWidth)
    }
}

public enum QwenImageEditVisionFeatureEncoder {
    public static func encode(
        modelPath: URL,
        sourceImage: URL,
        plan: QwenImageEditPreprocessPlan
    ) throws -> QwenImageEditVisionFeatures {
        let store = MFluxStore(try WeightLoader.load(from: modelPath))
        let vision = try QwenImageEditVisionTransformer(store: store)
        let input = try QwenImageEditPreprocessor.visionInput(sourceImage: sourceImage, plan: plan)
        let features = vision(pixelValues: input.pixelValues, imageGridTHW: input.imageGridTHW)
        eval(features)
        let result = QwenImageEditVisionFeatures(
            imageFeatures: features,
            imageGridTHW: input.imageGridTHW)
        try result.validateMatches(promptImageTokenCount: input.imageTokenCount)
        return result
    }
}

public struct QwenImageEditVisionLanguageEncoding {
    public let features: QwenImageEditVisionFeatures
    public let tokens: QwenImageEditPromptTokens
    public let promptEmbeddings: QwenImageEditPromptEmbeddings
}

public enum QwenImageEditPromptImageEncoder {
    public static func encode(
        modelPath: URL,
        sourceImage: URL,
        prompt: String,
        plan: QwenImageEditPreprocessPlan
    ) async throws -> QwenImageEditVisionLanguageEncoding {
        let store = MFluxStore(try WeightLoader.load(from: modelPath))
        return try await encode(
            store: store,
            modelPath: modelPath,
            sourceImage: sourceImage,
            prompt: prompt,
            plan: plan)
    }

    static func encode(
        store: MFluxStore,
        modelPath: URL,
        sourceImage: URL,
        prompt: String,
        plan: QwenImageEditPreprocessPlan
    ) async throws -> QwenImageEditVisionLanguageEncoding {
        let vision = try QwenImageEditVisionTransformer(store: store)
        let text = try QwenTextEncoder(store: store, dropIdx: QwenImageEditPreprocessor.editTemplateStartIndex)
        let tokenizer = try await QwenImageEditPromptTokenizer(modelPath: modelPath)
        let visionInput = try QwenImageEditPreprocessor.visionInput(sourceImage: sourceImage, plan: plan)
        let imageFeatures = vision(
            pixelValues: visionInput.pixelValues,
            imageGridTHW: visionInput.imageGridTHW)
        eval(imageFeatures)
        let features = QwenImageEditVisionFeatures(
            imageFeatures: imageFeatures,
            imageGridTHW: visionInput.imageGridTHW)
        try features.validateMatches(promptImageTokenCount: visionInput.imageTokenCount)

        let promptInput = try QwenImageEditPreprocessor.visionLanguagePrompt(
            prompt: prompt,
            imageTokenCounts: [features.imageTokenCount])
        let tokens = tokenizer.tokenize(promptInput)
        try features.validateMatches(promptImageTokenCount: tokens.imageTokenCount)
        let promptEmbeddings = try text.encodeVisionLanguage(
            inputIDs: tokens.inputIDs,
            attentionMask: tokens.attentionMask,
            imageFeatures: features,
            templateDropIndex: tokens.templateDropIndex)
        eval(promptEmbeddings.promptEmbeds, promptEmbeddings.attentionMask)
        return QwenImageEditVisionLanguageEncoding(
            features: features,
            tokens: tokens,
            promptEmbeddings: promptEmbeddings)
    }
}

public enum QwenImageEditDenoiseProbe {
    public static func predictVelocity(
        modelPath: URL,
        sourceImage: URL,
        prompt: String,
        plan: QwenImageEditPreprocessPlan,
        seed: UInt64?
    ) async throws -> QwenImageEditDenoiseResult {
        let store = MFluxStore(try WeightLoader.load(from: modelPath))
        let promptEncoding = try await QwenImageEditPromptImageEncoder.encode(
            store: store,
            modelPath: modelPath,
            sourceImage: sourceImage,
            prompt: prompt,
            plan: plan)
        let conditioning = try QwenImageEditConditioner.encode(
            store: store,
            sourceImage: sourceImage,
            plan: plan)

        let targetRows = plan.outputHeight / 16
        let targetColumns = plan.outputWidth / 16
        let targetCount = targetRows * targetColumns
        if let seed { MLXRandom.seed(seed) }
        let targetLatents = MLXRandom.normal([1, targetCount, 64]).asType(.float32)
        let inputs = try QwenImageEditTransformerInputs(
            targetLatents: targetLatents,
            targetPatchRows: targetRows,
            targetPatchColumns: targetColumns,
            conditioning: conditioning)
        let transformer = try QwenTransformer(store: store)
        let scheduler = FlowMatchEulerScheduler(steps: plan.steps, imageSeqLen: targetCount)
        let imageShapes = inputs.imageShapes.map {
            (frame: $0.frame, height: $0.height, width: $0.width)
        }
        let velocity = transformer(
            latents: inputs.hiddenStates,
            promptEmbeds: promptEncoding.promptEmbeddings.promptEmbeds,
            timestep: scheduler.sigmas[0],
            imageShapes: imageShapes)
        eval(velocity)
        return try QwenImageEditDenoiseResult(
            combinedVelocity: velocity,
            targetLatentCount: inputs.targetLatentCount,
            imageShapes: inputs.imageShapes)
    }
}

final class QwenImageEditPipeline {
    private let modelPath: URL
    private let store: MFluxStore
    private let transformer: QwenTransformer
    private let decoder: Qwen3DVAEDecoder

    init(modelPath: URL) throws {
        self.modelPath = modelPath
        self.store = MFluxStore(try WeightLoader.load(from: modelPath))
        self.transformer = try QwenTransformer(store: store)
        self.decoder = try Qwen3DVAEDecoder(store: store)
    }

    func edit(
        prompt: String,
        sourceImage: URL,
        width: Int?,
        height: Int?,
        steps: Int,
        guidance: Float,
        seed: UInt64?,
        progress: (Int, Int, Double?) -> Void
    ) async throws -> MLXArray {
        let plan = try QwenImageEditPreprocessPlan(
            sourceImage: sourceImage,
            requestedWidth: width,
            requestedHeight: height,
            steps: steps,
            guidance: guidance)
        let positive = try await QwenImageEditPromptImageEncoder.encode(
            store: store,
            modelPath: modelPath,
            sourceImage: sourceImage,
            prompt: prompt,
            plan: plan)
        let negative = try await QwenImageEditPromptImageEncoder.encode(
            store: store,
            modelPath: modelPath,
            sourceImage: sourceImage,
            prompt: "",
            plan: plan)
        let conditioning = try QwenImageEditConditioner.encode(
            store: store,
            sourceImage: sourceImage,
            plan: plan)

        let targetRows = plan.outputHeight / 16
        let targetColumns = plan.outputWidth / 16
        let targetCount = targetRows * targetColumns
        if let seed { MLXRandom.seed(seed) }
        var latents = MLXRandom.normal([1, targetCount, 64]).asType(.float32)
        let scheduler = FlowMatchEulerScheduler(steps: plan.steps, imageSeqLen: targetCount)

        let start = Date()
        for step in 0 ..< plan.steps {
            let inputs = try QwenImageEditTransformerInputs(
                targetLatents: latents,
                targetPatchRows: targetRows,
                targetPatchColumns: targetColumns,
                conditioning: conditioning)
            let imageShapes = inputs.imageShapes.map {
                (frame: $0.frame, height: $0.height, width: $0.width)
            }
            let timestep = scheduler.sigmas[step]
            let positiveCombined = transformer(
                latents: inputs.hiddenStates,
                promptEmbeds: positive.promptEmbeddings.promptEmbeds,
                timestep: timestep,
                imageShapes: imageShapes)
            let negativeCombined = transformer(
                latents: inputs.hiddenStates,
                promptEmbeds: negative.promptEmbeddings.promptEmbeds,
                timestep: timestep,
                imageShapes: imageShapes)
            let positiveVelocity = try QwenImageEditDenoiseResult(
                combinedVelocity: positiveCombined,
                targetLatentCount: inputs.targetLatentCount,
                imageShapes: inputs.imageShapes).targetVelocity
            let negativeVelocity = try QwenImageEditDenoiseResult(
                combinedVelocity: negativeCombined,
                targetLatentCount: inputs.targetLatentCount,
                imageShapes: inputs.imageShapes).targetVelocity
            let guided = QwenGuidance.computeGuidedNoise(
                positive: positiveVelocity,
                negative: negativeVelocity,
                guidance: plan.guidance)
            latents = scheduler.step(latent: latents, velocity: guided, stepIndex: step)
            eval(latents)
            let elapsed = Date().timeIntervalSince(start)
            progress(step + 1, plan.steps, elapsed / Double(step + 1) * Double(plan.steps - step - 1))
        }

        let unpacked = Self.unpackTargetLatents(
            latents,
            targetPatchRows: targetRows,
            targetPatchColumns: targetColumns)
        eval(unpacked)
        let image = decoder.decode(unpacked)
        eval(image)
        return image
    }

    static func unpackTargetLatents(
        _ latents: MLXArray,
        targetPatchRows: Int,
        targetPatchColumns: Int
    ) -> MLXArray {
        latents
            .reshaped([1, targetPatchRows, targetPatchColumns, 16, 2, 2])
            .transposed(0, 3, 1, 4, 2, 5)
            .reshaped([1, 16, targetPatchRows * 2, targetPatchColumns * 2])
    }
}

final class QwenImageEditVisionTransformer {
    private static let patchSize = 14
    private static let temporalPatchSize = 2
    private static let inChannels = 3
    private static let embedDim = 1280
    private static let depth = 32
    private static let heads = 16
    private static let headDim = 80
    private static let mlpHidden = 3420
    private static let spatialMergeSize = 2
    private static let windowSize = 112
    private static let fullAttentionBlocks: Set<Int> = [7, 15, 23, 31]
    private static let spatialMergeUnit = spatialMergeSize * spatialMergeSize

    private let patchEmbed: QwenEditVisionPatchEmbed
    private let blocks: [QwenEditVisionBlock]
    private let merger: QwenEditPatchMerger

    init(store: MFluxStore) throws {
        patchEmbed = try QwenEditVisionPatchEmbed(store: store)
        blocks = try (0 ..< Self.depth).map { try QwenEditVisionBlock(store: store, index: $0) }
        merger = try QwenEditPatchMerger(store: store)
    }

    static func mergedTokenCount(imageGridTHW: [Int]) throws -> Int {
        guard imageGridTHW.count == 3 else {
            throw FluxError.invalidRequest("Qwen edit vision grid must be [t,h,w]")
        }
        let product = imageGridTHW.reduce(1, *)
        guard product > 0, product % spatialMergeUnit == 0 else {
            throw FluxError.invalidRequest("Qwen edit vision grid product must be positive and divisible by \(spatialMergeUnit)")
        }
        return product / spatialMergeUnit
    }

    func callAsFunction(pixelValues: MLXArray, imageGridTHW: [Int]) -> MLXArray {
        var hidden = patchEmbed(pixelValues)
        let (windowIndex, cuWindowSeqlens, cuSeqlens) = Self.windowIndexAndSeqlens(imageGridTHW: imageGridTHW)
        let (cos, sin) = Self.rotaryPositionEmbeddings(imageGridTHW: imageGridTHW, dtype: hidden.dtype)

        let seqLen = hidden.dim(0)
        let groupCount = seqLen / Self.spatialMergeUnit
        let indexArray = MLXArray(windowIndex)
        hidden = hidden.reshaped([groupCount, Self.spatialMergeUnit, Self.embedDim])
        hidden = hidden[indexArray, 0..., 0...].reshaped([seqLen, Self.embedDim])

        for (index, block) in blocks.enumerated() {
            let seqlens = Self.fullAttentionBlocks.contains(index) ? cuSeqlens : cuWindowSeqlens
            hidden = block(hidden, cos: cos, sin: sin, cuSeqlens: seqlens)
        }

        hidden = merger(hidden)
        let reverse = argSort(indexArray)
        return hidden[reverse, 0...]
    }

    private static func rotaryPositionEmbeddings(imageGridTHW: [Int], dtype: DType) -> (MLXArray, MLXArray) {
        let t = imageGridTHW[0]
        let h = imageGridTHW[1]
        let w = imageGridTHW[2]
        let mergeH = h / spatialMergeSize
        let mergeW = w / spatialMergeSize
        var positionIDs: [(Int, Int)] = []
        positionIDs.reserveCapacity(t * h * w)
        for _ in 0 ..< t {
            for blockH in 0 ..< mergeH {
                for blockW in 0 ..< mergeW {
                    for mergeHIndex in 0 ..< spatialMergeSize {
                        for mergeWIndex in 0 ..< spatialMergeSize {
                            positionIDs.append((
                                blockH * spatialMergeSize + mergeHIndex,
                                blockW * spatialMergeSize + mergeWIndex))
                        }
                    }
                }
            }
        }

        let rotaryDim = headDim / 2
        let maxGrid = max(h, w)
        let invFreq = stride(from: 0, to: rotaryDim, by: 2).map {
            Float(1) / pow(Float(10_000), Float($0) / Float(rotaryDim))
        }
        var cosValues: [Float] = []
        var sinValues: [Float] = []
        cosValues.reserveCapacity(positionIDs.count * headDim)
        sinValues.reserveCapacity(positionIDs.count * headDim)

        var table = [[Float]]()
        table.reserveCapacity(maxGrid)
        for pos in 0 ..< maxGrid {
            table.append(invFreq.map { Float(pos) * $0 })
        }

        for (row, column) in positionIDs {
            let rotary = table[row] + table[column]
            let emb = rotary + rotary
            cosValues += emb.map { Foundation.cos($0) }
            sinValues += emb.map { Foundation.sin($0) }
        }
        let shape = [positionIDs.count, headDim]
        return (
            MLXArray(cosValues, shape).asType(dtype),
            MLXArray(sinValues, shape).asType(dtype)
        )
    }

    private static func windowIndexAndSeqlens(imageGridTHW: [Int]) -> ([Int32], [Int], [Int]) {
        let t = imageGridTHW[0]
        let gridH = imageGridTHW[1]
        let gridW = imageGridTHW[2]
        let llmGridH = gridH / spatialMergeSize
        let llmGridW = gridW / spatialMergeSize
        let window = windowSize / patchSize / spatialMergeSize
        let padH = window - llmGridH % window
        let padW = window - llmGridW % window
        let paddedH = llmGridH + padH
        let paddedW = llmGridW + padW
        let numWindowsH = paddedH / window
        let numWindowsW = paddedW / window

        var padded = [Int](repeating: -100, count: t * paddedH * paddedW)
        for ti in 0 ..< t {
            for row in 0 ..< llmGridH {
                for column in 0 ..< llmGridW {
                    let value = ti * llmGridH * llmGridW + row * llmGridW + column
                    padded[ti * paddedH * paddedW + row * paddedW + column] = value
                }
            }
        }

        var windowIndex: [Int32] = []
        var cuWindowSeqlens = [0]
        for ti in 0 ..< t {
            for wh in 0 ..< numWindowsH {
                for ww in 0 ..< numWindowsW {
                    var count = 0
                    for row in 0 ..< window {
                        for column in 0 ..< window {
                            let sourceRow = wh * window + row
                            let sourceColumn = ww * window + column
                            let value = padded[ti * paddedH * paddedW + sourceRow * paddedW + sourceColumn]
                            if value != -100 {
                                windowIndex.append(Int32(value))
                                count += 1
                            }
                        }
                    }
                    cuWindowSeqlens.append(cuWindowSeqlens.last! + count * spatialMergeUnit)
                }
            }
        }

        var uniqueCuWindowSeqlens: [Int] = []
        for value in cuWindowSeqlens where uniqueCuWindowSeqlens.last != value {
            uniqueCuWindowSeqlens.append(value)
        }

        var cuSeqlens = [0]
        var offset = 0
        offset += t * gridH * gridW
        cuSeqlens.append(offset)
        return (windowIndex, uniqueCuWindowSeqlens, cuSeqlens)
    }
}

private final class QwenEditVisionPatchEmbed {
    private let weight: MLXArray

    init(store: MFluxStore) throws {
        let weight = try store.tensor("text_encoder", "encoder.visual.patch_embed.proj.weight")
        guard weight.shape == [1280, 2, 14, 14, 3] else {
            throw FluxError.invalidRequest("Qwen edit vision patch weight shape mismatch: \(weight.shape)")
        }
        self.weight = weight.reshaped([1280, 2 * 14 * 14 * 3])
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let batch = pixelValues.dim(0)
        let channelsLast = pixelValues
            .reshaped([batch, 3, 2, 14, 14])
            .transposed(0, 2, 3, 4, 1)
            .reshaped([batch, 2 * 14 * 14 * 3])
        return matmul(channelsLast, weight.T)
    }
}

private final class QwenEditVisionBlock {
    private let norm1: MFluxRMSNorm
    private let norm2: MFluxRMSNorm
    private let attention: QwenEditVisionAttention
    private let mlp: QwenEditVisionMLP

    init(store: MFluxStore, index: Int) throws {
        let prefix = "encoder.visual.blocks.\(index)"
        norm1 = try store.rmsNorm("text_encoder", "\(prefix).norm1", eps: 1e-6)
        norm2 = try store.rmsNorm("text_encoder", "\(prefix).norm2", eps: 1e-6)
        attention = try QwenEditVisionAttention(store: store, prefix: "\(prefix).attn")
        mlp = try QwenEditVisionMLP(store: store, prefix: "\(prefix).mlp")
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, cuSeqlens: [Int]) -> MLXArray {
        var h = x + attention(norm1(x), cos: cos, sin: sin, cuSeqlens: cuSeqlens)
        h = h + mlp(norm2(h))
        return h
    }
}

private final class QwenEditVisionAttention {
    private let qkv: MFluxLinear
    private let proj: MFluxLinear
    private let heads = 16
    private let headDim = 80
    private let embedDim = 1280

    init(store: MFluxStore, prefix: String) throws {
        qkv = try store.linear("text_encoder", "\(prefix).qkv", inputDimensions: embedDim, outputDimensions: 3 * embedDim, bias: true)
        proj = try store.linear("text_encoder", "\(prefix).proj", inputDimensions: embedDim, outputDimensions: embedDim, bias: true)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, cuSeqlens: [Int]) -> MLXArray {
        let seq = x.dim(0)
        let qkvStates = qkv(x).reshaped([seq, 3, heads, headDim])
        var q = qkvStates[0..., 0, 0..., 0...].transposed(1, 0, 2)
        var k = qkvStates[0..., 1, 0..., 0...].transposed(1, 0, 2)
        let v = qkvStates[0..., 2, 0..., 0...].transposed(1, 0, 2)
        q = applyRope(q, cos: cos, sin: sin)
        k = applyRope(k, cos: cos, sin: sin)

        let scale = Float(1.0 / sqrt(Double(headDim)))
        let attended: MLXArray
        if cuSeqlens.count > 2 {
            var chunks: [MLXArray] = []
            for index in 0 ..< (cuSeqlens.count - 1) {
                let start = cuSeqlens[index]
                let end = cuSeqlens[index + 1]
                guard end > start else { continue }
                let range = start ..< end
                let qChunk = q[0..., range, 0...].expandedDimensions(axis: 0)
                let kChunk = k[0..., range, 0...].expandedDimensions(axis: 0)
                let vChunk = v[0..., range, 0...].expandedDimensions(axis: 0)
                let out = MLX.scaledDotProductAttention(
                    queries: qChunk,
                    keys: kChunk,
                    values: vChunk,
                    scale: scale,
                    mask: nil)
                chunks.append(out.squeezed(axis: 0))
            }
            attended = concatenated(chunks, axis: 1)
        } else {
            attended = MLX.scaledDotProductAttention(
                queries: q.expandedDimensions(axis: 0),
                keys: k.expandedDimensions(axis: 0),
                values: v.expandedDimensions(axis: 0),
                scale: scale,
                mask: nil
            ).squeezed(axis: 0)
        }

        let merged = attended.transposed(1, 0, 2).reshaped([seq, embedDim])
        return proj(merged)
    }

    private func applyRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let c = cos.reshaped([1, cos.dim(0), cos.dim(1)]).asType(.float32)
        let s = sin.reshaped([1, sin.dim(0), sin.dim(1)]).asType(.float32)
        let half = x.dim(-1) / 2
        let x1 = xf[.ellipsis, 0 ..< half]
        let x2 = xf[.ellipsis, half ..< x.dim(-1)]
        let rotated = concatenated([-x2, x1], axis: -1)
        return (xf * c + rotated * s).asType(x.dtype)
    }
}

private final class QwenEditVisionMLP {
    private let gate: MFluxLinear
    private let up: MFluxLinear
    private let down: MFluxLinear

    init(store: MFluxStore, prefix: String) throws {
        gate = try store.linear("text_encoder", "\(prefix).gate_proj", inputDimensions: 1280, outputDimensions: 3420, bias: true)
        up = try store.linear("text_encoder", "\(prefix).up_proj", inputDimensions: 1280, outputDimensions: 3420, bias: true)
        down = try store.linear("text_encoder", "\(prefix).down_proj", inputDimensions: 3420, outputDimensions: 1280, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private final class QwenEditPatchMerger {
    private let ln: MFluxRMSNorm
    private let mlp0: MFluxLinear
    private let mlp1: MFluxLinear

    init(store: MFluxStore) throws {
        ln = try store.rmsNorm("text_encoder", "encoder.visual.merger.ln_q", eps: 1e-6)
        mlp0 = try store.linear("text_encoder", "encoder.visual.merger.mlp_0", inputDimensions: 5120, outputDimensions: 5120, bias: true)
        mlp1 = try store.linear("text_encoder", "encoder.visual.merger.mlp_1", inputDimensions: 5120, outputDimensions: QwenTextEncoder.hidden, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = ln(x)
        let merged = normed.reshaped([-1, 5120])
        return mlp1(gelu(mlp0(merged)))
    }
}
