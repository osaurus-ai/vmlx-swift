import Foundation
import MLX
import MLXLMCommon

/// Qwen3.5 positions walk timestamp-separated video frame groups, not one T-frame block.
/// Pure integer implementation is also used by fixture/regression tests.
public enum D3MRoPE {
    public static func positions(
        tokens: [Int], images: [THW], videos: [THW],
        merge: Int, imageToken: Int, videoToken: Int
    ) throws -> [[Int32]] {
        guard merge > 0 else { throw DecisionRuntimeError.invalid("Invalid vision merge size") }
        if images.isEmpty && videos.isEmpty {
            return Array(repeating: tokens.indices.map(Int32.init), count: 3)
        }
        let grids = videos.flatMap { grid in (0 ..< grid.t).map { _ in THW(1, grid.h, grid.w) } }
        var imageIndex = 0
        var videoIndex = 0
        var at = 0
        var current = 0
        var axes = Array(repeating: [Int32](), count: 3)
        while at < tokens.count {
            let token = tokens[at]
            if token != imageToken && token != videoToken {
                for axis in 0 ..< 3 { axes[axis].append(Int32(current)) }
                current += 1
                at += 1
                continue
            }
            let grid: THW
            if token == imageToken {
                guard imageIndex < images.count else {
                    throw DecisionRuntimeError.invalid("Missing image position grid")
                }
                grid = images[imageIndex]
                imageIndex += 1
            } else {
                guard videoIndex < grids.count else {
                    throw DecisionRuntimeError.invalid("Missing video-group position grid")
                }
                grid = grids[videoIndex]
                videoIndex += 1
            }
            let h = grid.h / merge
            let w = grid.w / merge
            let count = grid.t * h * w
            guard h > 0, w > 0, grid.h % merge == 0, grid.w % merge == 0,
                at + count <= tokens.count, tokens[at ..< (at + count)].allSatisfy({ $0 == token })
            else {
                throw DecisionRuntimeError.invalid(
                    "Vision token count disagrees with position grid")
            }
            for t in 0 ..< grid.t {
                for y in 0 ..< h {
                    for x in 0 ..< w {
                        axes[0].append(Int32(current + t))
                        axes[1].append(Int32(current + y))
                        axes[2].append(Int32(current + x))
                    }
                }
            }
            current += max(grid.h, grid.w) / merge
            at += count
        }
        guard imageIndex == images.count, videoIndex == grids.count else {
            throw DecisionRuntimeError.invalid("Unconsumed vision position grids")
        }
        return axes
    }
}
