// SPDX-License-Identifier: MIT (port). SeedVR2 model/weights: Apache-2.0 / ByteDance.
// Port of mflux seedvr2_transformer/window.py (shift == false path).
//
// Builds gather/scatter indices that reorder video tokens into variable-size windows.
// All integer logic — produced once per forward from the [t,h,w] grid shape.
import Foundation
import MLX

public struct WindowPartitioner {
    public let forwardIdx: MLXArray  // gather original tokens -> windowed order
    public let reverseIdx: MLXArray  // scatter back
    public let windowShapes: [[Int]]  // [t,h,w] per window
    public let windowCounts: [Int]  // windows per batch element

    /// vidShape: [[t,h,w]] per batch element. window: target (nt,nh,nw).
    public init(vidShape: [[Int]], window: [Int], shift: Bool = false) {
        var forward: [Int32] = []
        var shapes: [[Int]] = []
        var counts: [Int] = []
        var base = 0
        for s in vidShape {
            let (t, h, w) = (s[0], s[1], s[2])
            let wins = Self.makeWindows(t: t, h: h, w: w, num: window, shift: shift)
            counts.append(wins.count)
            for win in wins {
                let (t0, t1, h0, h1, w0, w1) = win
                shapes.append([t1 - t0, h1 - h0, w1 - w0])
                for tt in t0 ..< t1 {
                    for hh in h0 ..< h1 {
                        for ww in w0 ..< w1 {
                            forward.append(Int32(base + tt * (h * w) + hh * w + ww))
                        }
                    }
                }
            }
            base += t * h * w
        }
        self.forwardIdx = MLXArray(forward)
        self.windowShapes = shapes
        self.windowCounts = counts
        // reverse = argsort(forward)
        self.reverseIdx = argSort(MLXArray(forward), axis: 0)
    }

    public func partition(_ x: MLXArray) -> MLXArray { x[forwardIdx] }
    public func reverse(_ x: MLXArray) -> MLXArray { x[reverseIdx] }

    /// Returns window bounds (t0,t1,h0,h1,w0,w1), iterating iw (outer) → ih → it (inner)
    /// to match mflux append order exactly. Handles shift (half-window offset) windows.
    static func makeWindows(t: Int, h: Int, w: Int, num: [Int], shift: Bool = false) -> [(
        Int, Int, Int, Int, Int, Int
    )] {
        let (rnt, rnh, rnw) = (num[0], num[1], num[2])
        let scale = (Double(45 * 80) / Double(h * w)).squareRoot()
        let resizedH = Int((Double(h) * scale).rounded(.toNearestOrEven))
        let resizedW = Int((Double(w) * scale).rounded(.toNearestOrEven))
        let wh = ceilDiv(resizedH, rnh)
        let ww = ceilDiv(resizedW, rnw)
        let wt = ceilDiv(min(t, 30), rnt)

        let st: Double
        let sh: Double
        let sw: Double
        let nt: Int
        let nh: Int
        let nw: Int
        if shift {
            st = wt < t ? 0.5 : 0
            sh = wh < h ? 0.5 : 0
            sw = ww < w ? 0.5 : 0
            nt = st > 0 ? ceilDiv(Int((Double(t) - st).rounded(.up)), wt) + 1 : 1
            nh = sh > 0 ? ceilDiv(Int((Double(h) - sh).rounded(.up)), wh) + 1 : 1
            nw = sw > 0 ? ceilDiv(Int((Double(w) - sw).rounded(.up)), ww) + 1 : 1
        } else {
            st = 0
            sh = 0
            sw = 0
            nt = ceilDiv(t, wt)
            nh = ceilDiv(h, wh)
            nw = ceilDiv(w, ww)
        }

        var out: [(Int, Int, Int, Int, Int, Int)] = []
        for iw in 0 ..< nw {
            let w0 = max(Int((Double(iw) - sw) * Double(ww)), 0)
            let w1 = min(Int((Double(iw) - sw + 1) * Double(ww)), w)
            if w1 <= w0 { continue }
            for ih in 0 ..< nh {
                let h0 = max(Int((Double(ih) - sh) * Double(wh)), 0)
                let h1 = min(Int((Double(ih) - sh + 1) * Double(wh)), h)
                if h1 <= h0 { continue }
                for it in 0 ..< nt {
                    let t0 = max(Int((Double(it) - st) * Double(wt)), 0)
                    let t1 = min(Int((Double(it) - st + 1) * Double(wt)), t)
                    if t1 <= t0 { continue }
                    out.append((t0, t1, h0, h1, w0, w1))
                }
            }
        }
        return out
    }

    static func ceilDiv(_ a: Int, _ b: Int) -> Int { (a + b - 1) / b }
}
