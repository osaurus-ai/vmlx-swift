// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation

/// Minimal MIL (Core ML program 1.3) text + weight-blob builder for the ops
/// the ANE drafter needs. Emits exactly the syntax `coremlc` renders for an
/// iOS18 program, which is what the private ANE compiler consumes.
///
/// Weights go into a MILBlob-format `weight.bin` (64-byte file header, then
/// 64-byte-aligned chunks each with a 64-byte metadata header carrying the
/// sentinel, dtype tag, size and absolute data offset).
public struct ANEMILBuilder {
    public enum DType: String { case fp16, int8, int32 }

    /// A typed SSA value in the program.
    public struct Value {
        public let name: String
        public let shape: [Int]
        public let dtype: DType
        var ref: String { name }
    }

    private var lines: [String] = []
    private var blob = Data()
    private var counter = 0
    private let functionName: String
    private var inputs: [Value] = []

    public init(function: String = "main") {
        functionName = function
        // MILBlob file header: version 1, count 2 (what coremlc writes).
        var header = [UInt8](repeating: 0, count: 64)
        header[0] = 0x01
        header[4] = 0x02
        blob.append(contentsOf: header)
    }

    // MARK: - blob

    private enum BlobType: UInt32 { case fp16 = 1, fp32 = 2, uint8 = 3, int8 = 4 }

    private mutating func blobAdd(_ bytes: UnsafeRawBufferPointer, type: BlobType) -> Int {
        let header = blob.count
        var meta = [UInt8](repeating: 0, count: 64)
        meta[0] = 0xEF; meta[1] = 0xBE; meta[2] = 0xAD; meta[3] = 0xDE
        withUnsafeBytes(of: type.rawValue) { meta.replaceSubrange(4 ..< 8, with: $0) }
        withUnsafeBytes(of: UInt64(bytes.count)) { meta.replaceSubrange(8 ..< 16, with: $0) }
        withUnsafeBytes(of: UInt64(header + 64)) { meta.replaceSubrange(16 ..< 24, with: $0) }
        blob.append(contentsOf: meta)
        blob.append(bytes.bindMemory(to: UInt8.self))
        let pad = (64 - bytes.count % 64) % 64
        if pad > 0 { blob.append(contentsOf: [UInt8](repeating: 0, count: pad)) }
        return header
    }

    public var weights: Data { blob }

    // MARK: - naming

    private mutating func fresh(_ base: String) -> String {
        counter += 1
        return "\(base)_\(counter)"
    }

    private static func shapeText(_ s: [Int]) -> String {
        "[" + s.map(String.init).joined(separator: ", ") + "]"
    }

    private static func tensorType(_ v: Value) -> String {
        "tensor<\(v.dtype.rawValue), \(shapeText(v.shape))>"
    }

    private static func product(_ s: [Int]) -> Int { s.reduce(1, *) }

    // MARK: - inputs / consts

    public mutating func input(_ name: String, shape: [Int]) -> Value {
        let v = Value(name: name, shape: shape, dtype: .fp16)
        inputs.append(v)
        return v
    }

    private mutating func constInt32(_ values: [Int], _ base: String) -> String {
        let n = fresh(base)
        lines.append("        tensor<int32, [\(values.count)]> \(n) = const()[name = string(\"\(n)\"), val = tensor<int32, [\(values.count)]>(\(Self.shapeText(values)))];")
        return n
    }

    private mutating func constScalarInt32(_ value: Int, _ base: String) -> String {
        let n = fresh(base)
        lines.append("        int32 \(n) = const()[name = string(\"\(n)\"), val = int32(\(value))];")
        return n
    }

    private mutating func constBool(_ value: Bool, _ base: String) -> String {
        let n = fresh(base)
        lines.append("        bool \(n) = const()[name = string(\"\(n)\"), val = bool(\(value))];")
        return n
    }

    private mutating func constString(_ value: String, _ base: String) -> String {
        let n = fresh(base)
        lines.append("        string \(n) = const()[name = string(\"\(n)\"), val = string(\"\(value)\")];")
        return n
    }

    private mutating func constScalarFP16(_ value: Float, _ base: String) -> String {
        let n = fresh(base)
        lines.append("        fp16 \(n) = const()[name = string(\"\(n)\"), val = fp16(\(Self.fp16Literal(value)))];")
        return n
    }

    private static func fp16Literal(_ value: Float) -> String {
        // Hex float keeps every fp16 bit exact through the text parser.
        String(format: "%a", Double(Float16(value)))
    }

    /// fp16 constant tensor from a blob chunk.
    public mutating func constFP16(_ values: [Float16], shape: [Int], _ base: String = "c") -> Value {
        precondition(values.count == Self.product(shape))
        let n = fresh(base)
        let off = values.withUnsafeBytes { blobAdd($0, type: .fp16) }
        lines.append("        tensor<fp16, \(Self.shapeText(shape))> \(n) = const()[name = string(\"\(n)\"), val = tensor<fp16, \(Self.shapeText(shape))>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(\(off))))];")
        return Value(name: n, shape: shape, dtype: .fp16)
    }

    /// int8 per-row affine weight [n, k, 1, 1] with fp16 row scales, ready for `conv`.
    public mutating func int8Weight(_ q: [Int8], scales: [Float16], n: Int, k: Int, _ base: String = "w") -> Value {
        precondition(q.count == n * k && scales.count == n)
        let name = fresh(base)
        let qOff = q.withUnsafeBytes { blobAdd($0, type: .int8) }
        let sOff = scales.withUnsafeBytes { blobAdd($0, type: .fp16) }
        lines.append("        tensor<fp16, [\(n), \(k), 1, 1]> \(name) = constexpr_affine_dequantize()[axis = int32(0), name = string(\"\(name)\"), quantized_data = tensor<int8, [\(n), \(k), 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(\(qOff)))), scale = tensor<fp16, [\(n)]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(\(sOff)))), zero_point = int8(0)];")
        return Value(name: name, shape: [n, k, 1, 1], dtype: .fp16)
    }

    // MARK: - ops

    private mutating func emit(_ base: String, shape: [Int], _ body: (String) -> String) -> Value {
        let n = fresh(base)
        lines.append("        tensor<fp16, \(Self.shapeText(shape))> \(n) = \(body(n))[name = string(\"\(n)\")];")
        return Value(name: n, shape: shape, dtype: .fp16)
    }

    /// 1x1 conv: x [B, k, H, W] × weight [n, k, 1, 1] → [B, n, H, W].
    public mutating func conv(_ x: Value, weight: Value, _ base: String = "conv") -> Value {
        let dil = constInt32([1, 1], "dil"), grp = constScalarInt32(1, "grp")
        let pad = constInt32([0, 0, 0, 0], "pad"), pt = constString("valid", "pt")
        let st = constInt32([1, 1], "st")
        var shape = x.shape
        shape[1] = weight.shape[0]
        return emit(base, shape: shape) { _ in
            "conv(dilations = \(dil), groups = \(grp), pad = \(pad), pad_type = \(pt), strides = \(st), weight = \(weight.ref), x = \(x.ref))"
        }
    }

    private static func broadcast(_ a: [Int], _ b: [Int]) -> [Int] {
        precondition(a.count == b.count)
        return zip(a, b).map { max($0, $1) }
    }

    public mutating func add(_ x: Value, _ y: Value, _ base: String = "add") -> Value {
        emit(base, shape: Self.broadcast(x.shape, y.shape)) { _ in "add(x = \(x.ref), y = \(y.ref))" }
    }

    public mutating func sub(_ x: Value, _ y: Value, _ base: String = "sub") -> Value {
        emit(base, shape: Self.broadcast(x.shape, y.shape)) { _ in "sub(x = \(x.ref), y = \(y.ref))" }
    }

    public mutating func mul(_ x: Value, _ y: Value, _ base: String = "mul") -> Value {
        emit(base, shape: Self.broadcast(x.shape, y.shape)) { _ in "mul(x = \(x.ref), y = \(y.ref))" }
    }

    public mutating func mul(_ x: Value, scalar: Float, _ base: String = "muls") -> Value {
        let s = constScalarFP16(scalar, "s")
        return emit(base, shape: x.shape) { _ in "mul(x = \(x.ref), y = \(s))" }
    }

    public mutating func add(_ x: Value, scalar: Float, _ base: String = "adds") -> Value {
        let s = constScalarFP16(scalar, "s")
        return emit(base, shape: x.shape) { _ in "add(x = \(x.ref), y = \(s))" }
    }

    public mutating func silu(_ x: Value, _ base: String = "silu") -> Value {
        emit(base, shape: x.shape) { _ in "silu(x = \(x.ref))" }
    }

    public mutating func sigmoid(_ x: Value, _ base: String = "sig") -> Value {
        emit(base, shape: x.shape) { _ in "sigmoid(x = \(x.ref))" }
    }

    public mutating func reduceMean(_ x: Value, axis: Int, _ base: String = "mean") -> Value {
        let axes = constInt32([axis], "axes"), kd = constBool(true, "kd")
        var shape = x.shape
        shape[axis] = 1
        return emit(base, shape: shape) { _ in "reduce_mean(axes = \(axes), keep_dims = \(kd), x = \(x.ref))" }
    }

    public mutating func reduceMax(_ x: Value, axis: Int, _ base: String = "max") -> Value {
        let axes = constInt32([axis], "axes"), kd = constBool(true, "kd")
        var shape = x.shape
        shape[axis] = 1
        return emit(base, shape: shape) { _ in "reduce_max(axes = \(axes), keep_dims = \(kd), x = \(x.ref))" }
    }

    public mutating func rsqrt(_ x: Value, epsilon: Float, _ base: String = "rsqrt") -> Value {
        let eps = constScalarFP16(epsilon, "eps")
        return emit(base, shape: x.shape) { _ in "rsqrt(epsilon = \(eps), x = \(x.ref))" }
    }

    public mutating func concat(_ values: [Value], axis: Int, _ base: String = "cat") -> Value {
        let ax = constScalarInt32(axis, "ax"), il = constBool(false, "il")
        var shape = values[0].shape
        shape[axis] = values.map { $0.shape[axis] }.reduce(0, +)
        let list = values.map(\.ref).joined(separator: ", ")
        return emit(base, shape: shape) { _ in "concat(axis = \(ax), interleave = \(il), values = (\(list)))" }
    }

    public mutating func slice(_ x: Value, begin: [Int], size: [Int], _ base: String = "slice") -> Value {
        let b = constInt32(begin, "b"), s = constInt32(size, "sz")
        let shape = zip(size, x.shape).enumerated().map { i, pair in pair.0 < 0 ? x.shape[i] - begin[i] : pair.0 }
        return emit(base, shape: shape) { _ in "slice_by_size(begin = \(b), size = \(s), x = \(x.ref))" }
    }

    public mutating func reshape(_ x: Value, _ shape: [Int], _ base: String = "rs") -> Value {
        precondition(Self.product(shape) == Self.product(x.shape))
        let sh = constInt32(shape, "sh")
        return emit(base, shape: shape) { _ in "reshape(shape = \(sh), x = \(x.ref))" }
    }

    public mutating func transpose(_ x: Value, perm: [Int], _ base: String = "tr") -> Value {
        let p = constInt32(perm, "perm")
        let shape = perm.map { x.shape[$0] }
        return emit(base, shape: shape) { _ in "transpose(perm = \(p), x = \(x.ref))" }
    }

    /// Batched matmul on the last two dims; leading dims broadcast.
    public mutating func matmul(_ x: Value, _ y: Value, _ base: String = "mm") -> Value {
        let tx = constBool(false, "tx"), ty = constBool(false, "ty")
        var shape = Self.broadcast(Array(x.shape.dropLast(2)) + [1, 1], Array(y.shape.dropLast(2)) + [1, 1])
        shape[shape.count - 2] = x.shape[x.shape.count - 2]
        shape[shape.count - 1] = y.shape[y.shape.count - 1]
        return emit(base, shape: shape) { _ in "matmul(transpose_x = \(tx), transpose_y = \(ty), x = \(x.ref), y = \(y.ref))" }
    }

    public mutating func softmax(_ x: Value, axis: Int, _ base: String = "sm") -> Value {
        let ax = constScalarInt32(axis, "ax")
        return emit(base, shape: x.shape) { _ in "softmax(axis = \(ax), x = \(x.ref))" }
    }

    public mutating func clip(_ x: Value, low: Float, high: Float, _ base: String = "clip") -> Value {
        let a = constScalarFP16(low, "lo"), b = constScalarFP16(high, "hi")
        return emit(base, shape: x.shape) { _ in "clip(alpha = \(a), beta = \(b), x = \(x.ref))" }
    }

    // MARK: - program text

    public func program(returning outputs: [Value]) -> String {
        var text = "program(1.3)\n"
        text += "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3520.4.1\"}, {\"coremlc-version\", \"3520.5.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"
        let params = inputs.map { "\(Self.tensorType($0)) \($0.name)" }.joined(separator: ", ")
        text += "    func \(functionName)<ios18>(\(params)) {\n"
        text += lines.joined(separator: "\n") + "\n"
        text += "        } -> (\(outputs.map(\.ref).joined(separator: ", ")));\n}\n"
        return text
    }
}
