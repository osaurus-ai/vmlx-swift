// Copyright © 2026 Osaurus. All rights reserved.

/// Shader-source generation only. One lane owns 16 consecutive coefficients.
/// The supported K%32 contract makes each lane chunk wholly valid or wholly absent.
/// Keep each coefficient FMA in order; packed-word reuse must not reassociate sums.
enum JANGHDecodeQDot {
    static func source(bits: Int, packed: String, rowBase: String, columnBase: String,
                       values: String, accumulator: String, alpha: Float, beta: Float) -> String {
        precondition([2, 3, 4, 6, 8].contains(bits))
        // `packed` may be constant or device storage in an MLX custom kernel.
        // Infer its address space instead of casting it to a device pointer.
        var lines = ["{", "uint qbase = ((\(columnBase)) * \(bits)u) >> 5u;", "auto qwords = \(packed) + \(rowBase) + qbase;"]
        let units = bits == 3 ? 2 : 16 * bits / 32
        for word in 0 ..< units { lines.append("uint qw\(word) = uint(qwords[\(word)]);") }
        let useLUT = bits <= 4
        if useLUT {
            let count = 1 << bits
            let center = Float(count - 1) / 2
            let table = (0 ..< count).map { code -> String in
                let u = Float(code) - center
                let level = u * alpha.addingProduct(beta, u * u)
                return "as_type<float>(0x\(String(level.bitPattern, radix: 16))u)"
            }.joined(separator: ", ")
            lines.append("const float LV[\(count)] = {\(table)};")
        }
        if bits == 3 {
            // A 16-value chunk starts at bit 0 or 16 of a UInt32 word and spans 48 bits.
            // Realign into (a0, a1) once per lane to eliminate 16 branches in the inner loop.
            lines.append("uint qstart = ((\(columnBase)) * 3u) & 31u;")
            lines.append("uint a0 = qstart == 0u ? qw0 : ((qw0 >> 16u) | (qw1 << 16u));")
            lines.append("uint a1 = qstart == 0u ? qw1 : (qw1 >> 16u);")
        }
        for i in 0 ..< 16 {
            let code: String
            if bits == 3 {
                let bit = i * 3
                if bit + 3 <= 32 {
                    code = bit == 0 ? "a0" : "(a0 >> \(bit)u)"
                } else if bit >= 32 {
                    code = bit == 32 ? "a1" : "(a1 >> \(bit - 32)u)"
                } else {
                    code = "((a0 >> \(bit)u) | (a1 << \(32 - bit)u))"
                }
            } else {
                let bit = i * bits, word = bit / 32, shift = bit % 32
                var unpacked = "(qw\(word) >> \(shift)u)"
                if shift + bits > 32 {
                    unpacked = "(\(unpacked) | (qw\(word + 1) << \(32 - shift)u))"
                }
                code = unpacked
            }
            let mask = (1 << bits) - 1
            if useLUT {
                lines.append("\(accumulator) = fma(\(values)[\(i)], LV[\(code) & \(mask)u], \(accumulator));")
            } else {
                lines.append("{ float u = float(\(code) & \(mask)u) - \(Float((1 << bits) - 1) / 2)f; float level = u * fma(\(beta)f, u * u, \(alpha)f); \(accumulator) = fma(\(values)[\(i)], level, \(accumulator)); }")
            }
        }
        lines.append("}")
        return lines.joined(separator: "\n")
    }
}
