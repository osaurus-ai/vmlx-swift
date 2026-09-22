// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

@preconcurrency import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLMCommon

struct MiMoV26ProcessorConfiguration: Decodable, Sendable {
    let vision: MiMoV26VisionConfiguration?
    let processor: Settings

    struct Settings: Decodable, Sendable {
        let image_min_pixels: Int
        let image_max_pixels: Int
        let video_min_pixels: Int
        let video_max_pixels: Int
        let video_total_max_pixels: Int
        let fps: Double
        let min_frames: Int?
        let max_frames: Int
        let num_frames: Int?
        let image_token_id: Int
        let video_token_id: Int
        let vision_start_token_id: Int
        let vision_end_token_id: Int
        let video_start_token_id: Int
        let video_end_token_id: Int
        let audio_start_token_id: Int
        let audio_end_token_id: Int
        let audio_token_id: Int
        let audio_sampling_rate: Int
        let audio_hop_length: Int
        let audio_stride_size: Int
        let audio_avg_pooler: Int
        let audio_group_size: Int
    }
    enum CodingKeys: String, CodingKey { case vision = "vision_config", processor = "processor_config" }
}

struct MiMoV26MessageGenerator: MessageGenerator {
    func generate(message: Chat.Message) -> Message {
        var result = defaultMessageDict(for: message)
        if let ordered = message.contentParts {
            result["content"] = ordered.map { part -> [String: String] in
                switch part {
                case .text(let text): return ["type": "text", "text": text]
                case .image: return ["type": "image"]
                case .video: return ["type": "video"]
                case .audio: return ["type": "audio"]
                }
            }
            return result
        }
        guard !message.images.isEmpty || !message.videos.isEmpty || !message.audios.isEmpty else { return result }
        var parts: [[String: String]] = []
        parts += message.images.map { _ in ["type": "image"] }
        parts += message.videos.map { _ in ["type": "video"] }
        parts += message.audios.map { _ in ["type": "audio"] }
        if !message.content.isEmpty { parts.append(["type": "text", "text": message.content]) }
        result["content"] = parts
        return result
    }
}

/// Native placeholders are expanded before cache lookup. The media payload
/// retains item boundaries, so lazy tower work never changes the token key.
struct MiMoV26Processor: UserInputProcessor {
    let configuration: MiMoV26ProcessorConfiguration
    let tokenizer: any Tokenizer

    private struct Visual {
        let pixels: MLXArray
        let grid: THW
        let replacement: [Int]
    }

    init(_ configuration: MiMoV26ProcessorConfiguration, tokenizer: any Tokenizer) throws {
        let p = configuration.processor
        guard p.audio_sampling_rate > 0, p.audio_hop_length > 0,
            p.audio_stride_size == 2, p.audio_avg_pooler == 2, p.audio_group_size == 4,
            p.fps.isFinite, p.fps > 0, p.max_frames > 0 else {
            throw VLMError.processing("Unsupported MiMo processor configuration")
        }
        self.configuration = configuration
        self.tokenizer = tokenizer
    }

    private func pixels(_ image: CIImage, processing: UserInput.Processing?) throws -> MLXArray {
        let image = MediaProcessing.apply(image, processing: processing).toSRGB()
        guard !image.extent.isEmpty, !image.extent.isInfinite else { throw VLMError.processing("Invalid MiMo image extent") }
        // CI's floating rendering is 0...1; native MiMo pixel math starts at 0...255.
        return image.asMLXArray(colorSpace: CGColorSpace(name: CGColorSpace.sRGB)) * 255
    }

    private func image(_ image: UserInput.Image, processing: UserInput.Processing?) throws -> Visual {
        guard let v = configuration.vision else { throw VLMError.processing("MiMo configuration has no vision tower") }
        let p = configuration.processor
        let raw = try pixels(image.asCIImage(), processing: processing)
        let (height, width) = try MiMoV26Pixels.targetSize(height: raw.dim(2), width: raw.dim(3),
            factor: v.patchSize * v.mergeSize, minimum: p.image_min_pixels, maximum: p.image_max_pixels)
        let (patches, grid) = try MiMoV26Pixels.patches(raw, height: height, width: width,
            patch: v.patchSize, merge: v.mergeSize, temporal: v.temporalPatchSize)
        let count = grid.product / (v.mergeSize * v.mergeSize)
        return Visual(pixels: patches, grid: grid,
            replacement: [p.vision_start_token_id] + Array(repeating: p.image_token_id, count: count) + [p.vision_end_token_id])
    }

    private func sampleIndices(total: Int, fps: Double) throws -> [Int] {
        let p = configuration.processor
        if let count = p.num_frames {
            guard count >= 2, count <= total, count.isMultiple(of: 2) else {
                throw VLMError.processing("Invalid explicit MiMo video frame count")
            }
            return (0..<count).map { Int(floor(Double($0) * Double(total - 1) / Double(count - 1))) }
        }
        return try MiMoV26Pixels.videoSamples(total: total, sourceFPS: fps, targetFPS: p.fps,
            minimum: p.min_frames ?? 8, maximum: p.max_frames)
    }

    private func videoFrames(_ video: UserInput.Video) async throws -> [UserInput.VideoFrame] {
        switch video {
        case .frames(let frames):
            guard frames.count >= 2 else { throw VLMError.processing("MiMo video needs at least two frames") }
            // In-memory frames are already sampled and retain their original timestamps.
            return frames
        case .url(let url):
            return try await videoFrames(.avAsset(AVURLAsset(url: url)))
        case .avAsset(let asset):
            guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw VLMError.noVideoTrackFound }
            let fps = Double(try await track.load(.nominalFrameRate))
            let seconds = try await asset.load(.duration).seconds
            guard seconds.isFinite, seconds > 0, fps.isFinite, fps > 0,
                seconds * fps < Double(Int.max) else { throw VLMError.videoNotDecodable }
            let indices = try sampleIndices(total: Int((seconds * fps).rounded()), fps: fps)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.requestedTimeToleranceBefore = .zero
            generator.requestedTimeToleranceAfter = .zero
            var frames: [UserInput.VideoFrame] = []
            for index in indices {
                try Task.checkCancellation()
                let time = CMTime(seconds: Double(index) / fps, preferredTimescale: 60_000)
                let (image, _) = try await generator.image(at: time)
                frames.append(.init(frame: CIImage(cgImage: image), timeStamp: time))
            }
            return frames
        }
    }

    private func video(_ video: UserInput.Video, processing: UserInput.Processing?) async throws -> Visual {
        guard let v = configuration.vision else { throw VLMError.processing("MiMo configuration has no vision tower") }
        let p = configuration.processor
        let frames = try await videoFrames(video)
        let arrays = try frames.map { try pixels($0.frame, processing: processing) }
        guard let first = arrays.first, arrays.allSatisfy({ $0.shape == first.shape }) else {
            throw VLMError.processing("MiMo video frame dimensions must agree")
        }
        let maximum = max(p.video_min_pixels,
            min(p.video_total_max_pixels * v.temporalPatchSize / arrays.count, p.video_max_pixels))
        let (height, width) = try MiMoV26Pixels.targetSize(height: first.dim(2), width: first.dim(3),
            factor: v.patchSize * v.mergeSize, minimum: p.video_min_pixels, maximum: maximum)
        let (patches, grid) = try MiMoV26Pixels.patches(concatenated(arrays), height: height, width: width,
            patch: v.patchSize, merge: v.mergeSize, temporal: v.temporalPatchSize)
        let count = grid.h * grid.w / (v.mergeSize * v.mergeSize)
        var replacement = [p.video_start_token_id]
        for index in 0..<grid.t {
            let time = frames[min(index * v.temporalPatchSize, frames.count - 1)].timeStamp.seconds
            // The native processor stores timestamps in Float32 before formatting.
            replacement += tokenizer.encode(text: try MiMoV26Pixels.timestamp(Double(Float(time))), addSpecialTokens: false)
            replacement += [p.vision_start_token_id] + Array(repeating: p.video_token_id, count: count) + [p.vision_end_token_id]
        }
        replacement.append(p.video_end_token_id)
        return Visual(pixels: patches, grid: grid, replacement: replacement)
    }

    /// Resample each decoded channel before mixing, preserving the native
    /// frontend's floating-point operation order.
    private func audioSamples(_ audio: UserInput.Audio) throws -> [Float] {
        let target = configuration.processor.audio_sampling_rate
        switch audio {
        case .samples(let samples, let rate), .preEncoded(let samples, let rate, _):
            return try MiMoV26AudioFeatures.resample(samples, from: rate, to: target)
        case .array(let samples, let rate):
            guard samples.ndim == 1 || (samples.ndim == 2 && samples.dim(0) == 1) else {
                throw VLMError.processing("MiMo audio arrays must contain mono PCM")
            }
            return try MiMoV26AudioFeatures.resample(samples.reshaped(-1).asArray(Float.self), from: rate, to: target)
        case .url(let url):
            let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
            guard file.length > 0, file.length <= Int64(UInt32.max),
                let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: UInt32(file.length)) else {
                throw VLMError.processing("Invalid MiMo audio file length or format")
            }
            try file.read(into: buffer)
            guard let data = buffer.floatChannelData, buffer.frameLength > 0 else { throw VLMError.processing("MiMo audio file has no PCM") }
            let channels = Int(buffer.format.channelCount)
            guard channels > 0 else { throw VLMError.processing("MiMo audio has no channels") }
            let resampled = try (0..<channels).map {
                try MiMoV26AudioFeatures.resample(Array(UnsafeBufferPointer(start: data[$0], count: Int(buffer.frameLength))),
                    from: Int(buffer.format.sampleRate), to: target)
            }
            return (0..<resampled[0].count).map { index in resampled.reduce(Float(0)) { $0 + $1[index] } / Float(channels) }
        }
    }

    static func expand(_ tokens: [Int], triples: [[Int]], replacements: [[[Int]]]) throws -> [Int] {
        var used = Array(repeating: 0, count: triples.count)
        var result: [Int] = [], cursor = 0
        while cursor < tokens.count {
            if cursor + 2 < tokens.count,
                let kind = triples.firstIndex(where: { tokens[cursor..<(cursor + 3)].elementsEqual($0) }) {
                guard used[kind] < replacements[kind].count else { throw VLMError.processing("MiMo media placeholder has no payload") }
                result += replacements[kind][used[kind]]
                used[kind] += 1
                cursor += 3
            } else {
                result.append(tokens[cursor]); cursor += 1
            }
        }
        guard used == replacements.map(\.count) else { throw VLMError.processing("MiMo media payload has no placeholder") }
        return result
    }

    func prepare(input: UserInput) async throws -> LMInput {
        let p = configuration.processor
        let messages = MiMoV26MessageGenerator().generate(from: input)
        let tokens = try tokenizer.applyChatTemplate(messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        let boundaries = canonicalChatCacheBoundaries(tokenizer: tokenizer, messages: messages, tools: input.tools,
            additionalContext: input.additionalContext, promptTokens: tokens)
        guard !input.images.isEmpty || !input.videos.isEmpty || !input.audios.isEmpty else {
            return LMInput(text: .init(tokens: MLXArray(tokens), tokenIds: tokens),
                cacheScopeSalt: cacheScopeSalt(from: input.additionalContext),
                cachePrefixTokenCounts: boundaries.all, cacheStablePrefixTokenCounts: boundaries.stable,
                toolSchemas: input.tools)
        }
        let images = try input.images.map { try image($0, processing: input.processing) }
        var videos: [Visual] = []
        for item in input.videos { videos.append(try await video(item, processing: input.processing)) }
        let audio = try input.audios.map { try audioSamples($0) }
        let replacements = audio.map { samples -> [Int] in
            let melFrames = 1 + samples.count / p.audio_hop_length
            let divisor = p.audio_stride_size * p.audio_avg_pooler * p.audio_group_size
            let count = (melFrames + divisor - 1) / divisor
            return [p.audio_start_token_id] + Array(repeating: p.audio_token_id, count: count) + [p.audio_end_token_id]
        }
        let triples = [[p.vision_start_token_id, p.image_token_id, p.vision_end_token_id],
                       [p.vision_start_token_id, p.video_token_id, p.vision_end_token_id],
                       [p.audio_start_token_id, p.audio_token_id, p.audio_end_token_id]]
        let expanded = try Self.expand(tokens, triples: triples,
            replacements: [images.map(\.replacement), videos.map(\.replacement), replacements])
        // Only pre-media boundaries retain their original offsets. The complete
        // expanded prompt itself is persisted by the normal runtime path.
        let firstMedia = tokens.indices.first { index in
            index + 2 < tokens.count && triples.contains { tokens[index..<(index + 3)].elementsEqual($0) }
        } ?? tokens.count
        return LMInput(text: .init(tokens: MLXArray(expanded), tokenIds: expanded),
            image: images.isEmpty ? nil : .init(pixels: concatenated(images.map(\.pixels)), frames: images.map(\.grid)),
            video: videos.isEmpty ? nil : .init(pixels: concatenated(videos.map(\.pixels)), frames: videos.map(\.grid)),
            audio: audio.isEmpty ? nil : .init(waveform: MLXArray(audio.flatMap { $0 }), sampleRate: p.audio_sampling_rate,
                                              clipSampleCounts: audio.map(\.count)),
            mediaTokenIds: [p.image_token_id, p.video_token_id, p.audio_token_id],
            cacheScopeSalt: cacheScopeSalt(from: input.additionalContext),
            cachePrefixTokenCounts: boundaries.all.filter { $0 <= firstMedia },
            cacheStablePrefixTokenCounts: boundaries.stable.filter { $0 <= firstMedia }, toolSchemas: input.tools)
    }
}
