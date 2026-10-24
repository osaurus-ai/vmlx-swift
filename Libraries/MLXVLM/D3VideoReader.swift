import AVFoundation
import CoreImage
import Foundation
import MLXLMCommon

/// Vendor sampling: 2 fps, 4...32 frames distributed over the complete clip, with an even tail.
/// The processor consumes these frames without sampling them a second time.
enum D3VideoReader {
    static func read(_ url: URL) async throws -> sending UserInput.Video {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VLMError.noVideoTrackFound
        }
        let duration = try await asset.load(.duration).seconds
        let nominal = Double(try await track.load(.nominalFrameRate))
        let fps = nominal.isFinite && nominal >= 0.1 && nominal <= 1000 ? nominal : 24
        guard duration.isFinite, duration > 0 else { throw VLMError.videoNotDecodable }
        let total = max(1, Int((duration * fps).rounded()))
        let count = min(total, min(32, max(4, Int(duration * 2))))
        let indices = (0 ..< count).map { i in
            count == 1
                ? 0
                : Int((Double(i) * Double(total - 1) / Double(count - 1)).rounded(.toNearestOrEven))
        }
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        var frames: [UserInput.VideoFrame] = []
        for index in indices {
            try Task.checkCancellation()
            let time = CMTime(seconds: Double(index) / fps, preferredTimescale: 600_000)
            let (image, _) = try await generator.image(at: time)
            frames.append(.init(frame: CIImage(cgImage: image), timeStamp: time))
        }
        if frames.count % 2 != 0, let last = frames.last { frames.append(last) }
        return .frames(frames)
    }
}
