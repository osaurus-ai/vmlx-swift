import Foundation

/// Vendor d3_compose geometry: 128 bin centres, ordered edges and deterministic R4 IoU confidence.
public enum D3Box {
    public static let edgeNames = ["top", "bottom", "left", "right"]
    public static func suffix(presence: Bool) -> String {
        ((presence ? ["visible"] : []) + edgeNames).map { $0 + ":" }.joined(separator: "\n")
    }
    public static func user(
        state: DecisionJSON, instructions: String, codes: [String], presence: Bool,
        at seconds: Double? = nil
    ) -> String {
        let when =
            seconds.map {
                " in the video frame at "
                    + String(format: "%.1f", locale: Locale(identifier: "en_US_POSIX"), $0)
                    + " seconds"
            } ?? ""
        var lines = [
            "State:", D3Prompt.stateText(state), "", "Question:", instructions,
            "", "Box:",
            "Find the target\(when). Give its top, bottom, left and right edges, each as a code on a 128-step scale across the image: \(codes[0]) is the top or left side, \(codes[127]) the bottom or right side.",
        ]
        if presence {
            lines.append(
                "Also say whether the target is visible then: \(codes[0]) no, \(codes[1]) yes.")
        }
        lines += ["", "Reply with the edge codes."]
        return lines.joined(separator: "\n")
    }
    public static func decode(edges: [[Float]]) throws -> [String: Double] {
        guard edges.count == 4,
            edges.allSatisfy({
                $0.count == 128 && $0.allSatisfy({ $0.isFinite && $0 >= 0 }) && $0.reduce(0, +) > 0
            })
        else {
            throw DecisionRuntimeError.invalid("Invalid box edge distributions")
        }
        let expected = edges.map { p in
            p.enumerated().reduce(0.0) { $0 + Double($1.element) * (Double($1.offset) + 0.5) / 128 }
                / Double(p.reduce(0, +))
        }
        func ordered(_ a: Double, _ b: Double) -> (Double, Double) {
            if a < b { return (a, b) }
            let mid = (a + b) / 2
            let half = 0.5 / 128.0
            return (max(0, mid - half), min(1, mid + half))
        }
        let (top, bottom) = ordered(expected[0], expected[1])
        let (left, right) = ordered(expected[2], expected[3])
        return ["top": top, "bottom": bottom, "left": left, "right": right]
    }
    public static func confidence(edges: [[Float]], box: [String: Double], calibration: [[Double]])
        -> Double
    {
        var phi = 1.0
        for _ in 0 ..< 64 { phi = pow(1 + phi, 0.2) }
        let alpha = (0 ..< 4).map {
            pow(1 / phi, Double($0 + 1)).truncatingRemainder(dividingBy: 1)
        }
        let cdfs = edges.map { row -> [Double] in
            let total = row.reduce(0.0) { $0 + Double($1) }
            var sum = 0.0
            return row.map {
                sum += Double($0) / total
                return sum
            }
        }
        var hits = 0
        for sample in 1 ... 512 {
            let draw = (0 ..< 4).map { axis -> Double in
                let point = (0.5 + alpha[axis] * Double(sample)).truncatingRemainder(dividingBy: 1)
                let bin = cdfs[axis].firstIndex(where: { $0 >= point }) ?? 127
                return (Double(bin) + 0.5) / 128
            }
            let (t, b, l, r) = (draw[0], draw[1], draw[2], draw[3])
            if t >= b || l >= r { continue }
            let inter =
                max(0, min(b, box["bottom"]!) - max(t, box["top"]!))
                * max(0, min(r, box["right"]!) - max(l, box["left"]!))
            let union =
                (b - t) * (r - l) + (box["bottom"]! - box["top"]!) * (box["right"]! - box["left"]!)
                - inter
            if union > 0 && inter / union >= 0.5 { hits += 1 }
        }
        let raw = Double(hits) / 512
        guard let first = calibration.first, let last = calibration.last else { return raw }
        if raw <= first[0] { return first[1] }
        for (a, b) in zip(calibration, calibration.dropFirst()) where raw <= b[0] {
            return a[0] == b[0] ? a[1] : a[1] + (b[1] - a[1]) * (raw - a[0]) / (b[0] - a[0])
        }
        return last[1]
    }
}
