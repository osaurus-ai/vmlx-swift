import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

/// The JANGH NAX gather prefill at the batch sizes a 4,096-token chunk produces (4,096 x top-8 = 32,768 expert-sorted
/// rows; the tile code once cast `M - row` to `short` before clamping, so rows of a tile with >= 32,768 rows after it
/// computed with a negative row count: Naive-N0.5 NaN at 4,096-token prefill chunks)
/// rows over 256 experts), against the same kernel on 4,096-row slices of the same sorted rows (rows are independent
/// given their expert): must be finite, deterministic and slice-invariant.
final class JANGHPrefillLargeBatchTests: XCTestCase {
    private func contract() throws -> JANGHFormatContract {
        var books: [String: [String: Any]] = [:]
        for bits in [2, 3, 4, 6, 8] {
            let count = 1 << bits
            let center = Double(count - 1) / 2
            books[String(bits)] = ["alpha": 0.25, "beta": 0.0,
                                   "levels": (0 ..< count).map { (Double($0) - center) * 0.25 }]
        }
        let root: [String: Any] = [
            "jangtq": ["version": 2, "packing": "lsb-bitstream", "scale_dtype": "float16",
                       "codebook_family": "odd-cubic", "rotation": "hadamard32", "codebooks": books],
            "quantization": [
                "model.layers.0.mlp.switch_mlp.gate_proj": ["mode": "jangtq2", "bits": 2, "rotation": "hadamard32"],
                "model.layers.0.mlp.switch_mlp.up_proj": ["mode": "jangtq2", "bits": 2, "rotation": "hadamard32"],
                "model.layers.0.mlp.switch_mlp.down_proj": ["mode": "jangtq2", "bits": 3, "rotation": "hadamard32"],
            ],
        ]
        return try JANGHFormatContract(configuration: JSONSerialization.data(withJSONObject: root))
    }

    func testLargeSortedBatchMatchesSlices() throws {
        try XCTSkipUnless(JANGHPrefillKernel.nativeBackend == .nax, "NAX prefill")
        let c = try contract()
        let kernel = try JANGHPrefillKernel(contract: c, module: "model.layers.0.mlp.switch_mlp.gate_proj",
                                            upModule: "model.layers.0.mlp.switch_mlp.up_proj")
        MLXRandom.seed(4)
        let e = 256, n = 2048, k = 4096, rows = 40001  // > 2^15 rows (short wrap) and not a multiple of 64
        let gp = MLXRandom.randInt(low: 0, high: Int32.max, [e, n, k * 2 / 32]).asType(.uint32)
        let up = MLXRandom.randInt(low: 0, high: Int32.max, [e, n, k * 2 / 32]).asType(.uint32)
        let gs = (MLXRandom.uniform(0.005 ..< 0.02, [e, n])).asType(.float16)
        let us = (MLXRandom.uniform(0.005 ..< 0.02, [e, n])).asType(.float16)
        let ids = take(MLXRandom.randInt(low: 0, high: Int32(e), [rows]).asType(.uint32),
                       argSort(MLXRandom.randInt(low: 0, high: Int32(e), [rows]).asType(.uint32)))
        let sorted = take(ids, argSort(ids))
        let x = MLXRandom.normal([rows, k]).asType(.bfloat16)
        eval(gp, up, gs, us, sorted, x)
        func run(_ lo: Int, _ hi: Int) throws -> MLXArray {
            try kernel.projectSorted(x[lo ..< hi], packed: gp, scales: gs, indices: sorted[lo ..< hi],
                                     upPacked: up, upScales: us, limit: nil)
        }
        let whole = try run(0, rows)
        let whole2 = try run(0, rows)
        let sliced = concatenated(try stride(from: 0, to: rows, by: 4096).map { try run($0, min($0 + 4096, rows)) }, axis: 0)
        eval(whole, whole2, sliced)
        func bad(_ a: MLXArray) -> Int32 { (1 - MLX.isFinite(a).asType(.int32)).sum().item(Int32.self) }
        let diff = abs(whole.asType(.float32) - sliced.asType(.float32)).max().item(Float.self)
        let rerun = abs(whole.asType(.float32) - whole2.asType(.float32)).max().item(Float.self)
        print("JANGH large prefill: non-finite whole \(bad(whole)) sliced \(bad(sliced)); max |whole - sliced| \(diff); max |rerun diff| \(rerun); max |whole| \(abs(whole.asType(.float32)).max().item(Float.self))")
        let rowDiff = abs(whole.asType(.float32) - sliced.asType(.float32)).max(axis: -1).asArray(Float.self)
        let idsHost = sorted.asArray(UInt32.self)
        let badRows = rowDiff.enumerated().filter { $0.element > 1e-3 }.map(\.offset)
        print("JANGH large prefill: \(badRows.count) differing rows; first 12: \(badRows.prefix(12).map { r in "r\(r)%64=\(r % 64) e\(idsHost[r]) segStart=\(r == 0 || idsHost[r - 1] != idsHost[r]) prevE=\(r > 0 ? idsHost[r - 1] : 0)" })")
        var segSizes: [Int] = []; var cnt = 1
        for i in 1 ..< idsHost.count { if idsHost[i] == idsHost[i - 1] { cnt += 1 } else { segSizes.append(cnt); cnt = 1 } }
        print("JANGH large prefill: segments \(segSizes.count) mean \(Double(rows) / Double(segSizes.count + 1)) max \(segSizes.max() ?? 0)")
        let small = try run(0, 64)
        for m in [8192, 16384, 24576] { eval(try run(0, m)); let t = try run(0, m)
            print("JANGH large prefill: M=\(m) rows0-31 vs small \(abs(t[0 ..< 32].asType(.float32) - small[0 ..< 32].asType(.float32)).max().item(Float.self))") }
        print("JANGH large prefill: rows 0-31 whole vs small \(abs(whole[0 ..< 32].asType(.float32) - small[0 ..< 32].asType(.float32)).max().item(Float.self)); sliced vs small \(abs(sliced[0 ..< 32].asType(.float32) - small[0 ..< 32].asType(.float32)).max().item(Float.self))")
        XCTAssertEqual(bad(whole), 0)
        XCTAssertEqual(bad(sliced), 0)
        XCTAssertEqual(diff, 0)
        XCTAssertEqual(rerun, 0)
    }
}
