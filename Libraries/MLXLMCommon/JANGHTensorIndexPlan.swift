import Foundation

/// Header-only admission for custom banks. This does not read tensor payloads,
/// construct modules, select an architecture, or enable a model factory.
struct JANGHTensorIndexPlan: Sendable {
    struct Dimensions: Sendable {
        let experts: Int
        let input: Int
        let output: Int
    }

    struct TensorHeader: Sendable {
        let dtype: String
        let shape: [Int]
        /// Safetensors offsets relative to the start of the data section.
        let start: Int
        let end: Int
    }

    struct ShardHeader: Sendable {
        let fileBytes: Int
        let dataStart: Int
        let tensors: [String: TensorHeader]
    }

    struct Location: Sendable {
        let shard: String
        let tensor: String
        let fileOffset: Int
        let byteCount: Int
    }

    struct Banks: Sendable {
        let packed: Location
        let scales: Location
    }

    /// JANGT trellis tensors of one projection (see JANGTProjection for their meaning).
    struct TrellisBanks: Sendable {
        let packed: Location
        let scale: Location
        let signs: Location
        let outlierColumns: Location
        let outlierWeights: Location
        let columnScales: Location?
    }

    static let trellisSuffixes = [".jt_packed", ".jt_scale", ".jt_su", ".jt_ocol", ".jt_wout", ".jt_cs"]

    let projections: [String: Banks]
    let trellis: [String: TrellisBanks]

    init(
        contract: JANGHFormatContract, dimensions: [String: Dimensions],
        trellisDimensions: [String: Dimensions] = [:],
        weightMap: [String: String], shards: [String: ShardHeader]
    ) throws {
        typealias Failure = JANGHFormatContract.ValidationError
        guard Set(dimensions.keys) == Set(contract.projections.keys),
            Set(trellisDimensions.keys) == Set(contract.trellis.keys)
        else {
            throw Failure.invalid("JANGH architecture projection coverage mismatch")
        }
        // Every jt_* tensor must belong to an admitted trellis module; jt_cs exists only on down projections.
        let trellisNames = Set(contract.trellis.keys.flatMap { module in
            Self.trellisSuffixes.map { module + $0 }.filter {
                !$0.hasSuffix(".jt_cs") || module.hasSuffix(".down_proj")
            }
        })
        let expectedNames = Set(contract.projections.keys.flatMap {
            [$0 + ".tq2_packed", $0 + ".tq2_scales"]
        })
        func isCustom(_ name: String) -> Bool {
            name.hasSuffix(".tq2_packed") || name.hasSuffix(".tq2_scales")
        }
        func isTrellis(_ name: String) -> Bool { Self.trellisSuffixes.contains { name.hasSuffix($0) } }
        guard Set(weightMap.keys.filter(isCustom)) == expectedNames else {
            throw Failure.invalid("JANGH index projection coverage mismatch")
        }
        guard Set(weightMap.keys.filter(isTrellis)).isSubset(of: trellisNames) else {
            throw Failure.invalid("JANGT index has tensors outside the admitted trellis modules")
        }
        // Summaries must include every indexed shard, including ordinary weights,
        // so an unindexed or duplicate custom tensor cannot escape this audit.
        guard Set(weightMap.values) == Set(shards.keys) else {
            throw Failure.invalid("JANGH shard inventory mismatch")
        }
        var found = Set<String>()
        for (file, shard) in shards {
            guard !file.isEmpty, file != ".", file != "..",
                !file.contains("/"), !file.contains("\\"), file.hasSuffix(".safetensors"),
                shard.dataStart >= 8, shard.fileBytes >= shard.dataStart
            else { throw Failure.invalid("invalid JANGH shard summary") }
            var ranges: [(Int, Int)] = []
            for (name, header) in shard.tensors {
                guard header.start >= 0, header.end >= header.start,
                    header.end <= shard.fileBytes - shard.dataStart
                else { throw Failure.invalid("invalid safetensors byte range") }
                if header.end > header.start { ranges.append((header.start, header.end)) }
                if isCustom(name) {
                    guard expectedNames.contains(name), weightMap[name] == file,
                        found.insert(name).inserted
                    else { throw Failure.invalid("duplicate or unowned JANGH tensor") }
                }
                if isTrellis(name) {
                    guard trellisNames.contains(name), weightMap[name] == file else {
                        throw Failure.invalid("unowned or unindexed JANGT tensor")
                    }
                }
            }
            ranges.sort { $0.0 < $1.0 }
            for index in 1..<max(1, ranges.count) {
                guard ranges[index - 1].1 <= ranges[index].0 else {
                    throw Failure.invalid("overlapping safetensors byte ranges")
                }
            }
        }
        guard found == expectedNames else {
            throw Failure.invalid("missing indexed JANGH tensor")
        }

        func locate(_ name: String, bytesPerElement: Int) throws -> (TensorHeader, Location) {
            guard let file = weightMap[name], let shard = shards[file],
                let header = shard.tensors[name]
            else { throw Failure.invalid("missing JANGH bank") }
            var count = bytesPerElement
            for size in header.shape {
                let product = count.multipliedReportingOverflow(by: size)
                guard size > 0, !product.overflow else {
                    throw Failure.invalid("JANGH bank size overflow")
                }
                count = product.partialValue
            }
            // Range bounds above guarantee this sum cannot overflow.
            let offset = shard.dataStart + header.start
            guard header.end - header.start == count, offset.isMultiple(of: bytesPerElement) else {
                throw Failure.invalid("JANGH bank byte count or dtype alignment mismatch")
            }
            return (header, Location(shard: file, tensor: name, fileOffset: offset, byteCount: count))
        }
        var result: [String: Banks] = [:]
        for (module, d) in dimensions {
            for suffix in [".tq_packed", ".tq_norms", ".weight"] {
                let old = module + suffix
                guard weightMap[old] == nil, !shards.values.contains(where: { $0.tensors[old] != nil })
                else { throw Failure.invalid("conflicting JANGH and legacy weight representation") }
            }
            let (packed, p) = try locate(module + ".tq2_packed", bytesPerElement: 4)
            let (scales, s) = try locate(module + ".tq2_scales", bytesPerElement: 2)
            try contract.validateTensorHeaders(
                module: module, experts: d.experts, inputDimensions: d.input,
                outputDimensions: d.output, packedShape: packed.shape, packedDType: packed.dtype,
                scalesShape: scales.shape, scalesDType: scales.dtype)
            result[module] = Banks(packed: p, scales: s)
        }
        var trellisResult: [String: TrellisBanks] = [:]
        for (module, d) in trellisDimensions {
            for suffix in [".tq2_packed", ".tq2_scales", ".tq_packed", ".tq_norms", ".weight"] {
                guard weightMap[module + suffix] == nil else {
                    throw Failure.invalid("conflicting JANGT and other weight representation for \(module)")
                }
            }
            let k = contract.trellis[module]!
            let (packed, p) = try locate(module + ".jt_packed", bytesPerElement: 4)
            let (scale, sc) = try locate(module + ".jt_scale", bytesPerElement: 4)
            let (signs, su) = try locate(module + ".jt_su", bytesPerElement: 4)
            let (ocol, oc) = try locate(module + ".jt_ocol", bytesPerElement: 4)
            let (wout, wo) = try locate(module + ".jt_wout", bytesPerElement: 2)
            let outliers = ocol.shape.count == 2 ? ocol.shape[1] : 0
            guard d.input.isMultiple(of: 512), d.output.isMultiple(of: 8),
                packed.dtype == "U32",
                packed.shape == [d.experts, d.output, JANGTCode.wordsPerRow(inputs: d.input, k: k)],
                scale.dtype == "F32", scale.shape == [d.experts, d.output],
                signs.dtype == "F32", signs.shape == [d.input],
                ocol.dtype == "I32", outliers >= 1, outliers <= 32, ocol.shape == [d.experts, outliers],
                wout.dtype == "F16", wout.shape == [d.experts, d.output, outliers]
            else { throw Failure.invalid("JANGT tensor header mismatch for \(module)") }
            var cs: Location?
            if weightMap[module + ".jt_cs"] != nil {
                let (header, location) = try locate(module + ".jt_cs", bytesPerElement: 4)
                guard header.dtype == "F32", header.shape == [d.experts, d.input] else {
                    throw Failure.invalid("JANGT column-scale header mismatch for \(module)")
                }
                cs = location
            }
            trellisResult[module] = TrellisBanks(
                packed: p, scale: sc, signs: su, outlierColumns: oc, outlierWeights: wo, columnScales: cs)
        }
        projections = result
        trellis = trellisResult
    }
}
