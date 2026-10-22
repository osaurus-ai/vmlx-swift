import Cmlx
import Foundation
import MLX
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Owns packed banks in independent MLX allocations. Reads use leased source
/// descriptors, with no retained file mappings or dequantized weight copy.
final class JANGHMappedBanks {
    struct Projection {
        let packed: MLXArray
        let scales: MLXArray
    }
    let contract: JANGHFormatContract
    private let banks: [String: Projection]
    private let trellisBanks: [String: JANGTProjection]

    /// Retains verified descriptors across the metadata-to-mapping boundary.
    /// Replacing a pathname after this lease is acquired does not redirect reads.
    final class SourceLease {
        let metadata: JANGHHeaderAdapter.Snapshot
        let contract: JANGHFormatContract
        let plan: JANGHTensorIndexPlan
        fileprivate let index: FileHandle
        fileprivate let files: [String: FileHandle]

        init(directory: URL, metadata: JANGHHeaderAdapter.Snapshot,
             contract: JANGHFormatContract, dimensions: [String: JANGHTensorIndexPlan.Dimensions],
             trellisDimensions: [String: JANGHTensorIndexPlan.Dimensions] = [:]) throws {
            let plan = try JANGHTensorIndexPlan(contract: contract, dimensions: dimensions,
                                               trellisDimensions: trellisDimensions,
                                               weightMap: metadata.weightMap, shards: metadata.shards)
            guard directory.isFileURL else {
                throw JANGHFormatContract.ValidationError.invalid("JANGH mapping requires a file directory")
            }
            let root = directory.resolvingSymlinksInPath().standardizedFileURL
            let descriptor = open(root.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard descriptor >= 0 else {
                throw JANGHFormatContract.ValidationError.invalid("cannot open JANGH mapping directory")
            }
            defer { close(descriptor) }
            let index = try JANGHHeaderAdapter.openRegular(rootDescriptor: descriptor, name: metadata.indexName)
            var files: [String: FileHandle] = [:]
            do {
                guard try JANGHHeaderAdapter.identity(index) == metadata.indexIdentity else {
                    throw JANGHFormatContract.ValidationError.invalid("JANGH index identity changed before mapping")
                }
                for name in metadata.shards.keys.sorted() {
                    let file = try JANGHHeaderAdapter.openRegular(rootDescriptor: descriptor, name: name)
                    files[name] = file
                    guard let expected = metadata.identities[name],
                        try JANGHHeaderAdapter.identity(file) == expected
                    else { throw JANGHFormatContract.ValidationError.invalid("JANGH shard identity changed before mapping") }
                }
            } catch {
                try? index.close()
                files.values.forEach { try? $0.close() }
                throw error
            }
            self.metadata = metadata
            self.contract = contract
            self.plan = plan
            self.index = index
            self.files = files
        }

        /// Keeps the verified descriptor alive while an exact mapped region is created.
        /// Cached GPU views must also revalidate their source before each selection.
        func withValidatedFile<T>(
            for location: JANGHTensorIndexPlan.Location,
            _ body: (Int32, JANGHTensorIndexPlan.TensorHeader) throws -> T
        ) throws -> T {
            guard let file = files[location.shard],
                let expected = metadata.identities[location.shard],
                let header = metadata.shards[location.shard]?.tensors[location.tensor]
            else {
                throw JANGHFormatContract.ValidationError.invalid("unknown JANGH leased tensor")
            }
            func validate() throws {
                guard try JANGHHeaderAdapter.identity(index) == metadata.indexIdentity,
                    try JANGHHeaderAdapter.identity(file) == expected
                else {
                    throw JANGHFormatContract.ValidationError.invalid("JANGH source lease became stale")
                }
            }
            try validate()
            let result = try body(file.fileDescriptor, header)
            try validate()
            return result
        }

        deinit {
            try? index.close()
            files.values.forEach { try? $0.close() }
        }
    }

    convenience init(directory: URL, metadata: JANGHHeaderAdapter.Snapshot,
                     contract: JANGHFormatContract, dimensions: [String: JANGHTensorIndexPlan.Dimensions]) throws {
        try self.init(source: SourceLease(directory: directory, metadata: metadata,
                                        contract: contract, dimensions: dimensions))
    }

    init(source: SourceLease, modules: Set<String>? = nil) throws {
        defer { withExtendedLifetime(source) {} }
        let metadata = source.metadata
        let files = source.files
        let index = source.index
        let plan = source.plan
        if let modules {
            guard !modules.isEmpty, modules.isSubset(of: Set(plan.projections.keys)) else {
                throw JANGHFormatContract.ValidationError.invalid("invalid JANGH mapped module subset")
            }
        }
        for (name, file) in files {
            guard try JANGHHeaderAdapter.identity(file) == metadata.identities[name] else {
                throw JANGHFormatContract.ValidationError.invalid("JANGH source lease became stale before mapping")
            }
        }
        guard try JANGHHeaderAdapter.identity(index) == metadata.indexIdentity else {
            throw JANGHFormatContract.ValidationError.invalid("JANGH index lease became stale before mapping")
        }
        // Copy packed banks once into their final compute allocations.
        func read(_ location: JANGHTensorIndexPlan.Location) throws -> MLXArray {
            guard let file = files[location.shard],
                let header = metadata.shards[location.shard]?.tensors[location.tensor],
                header.shape.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) })
            else { throw JANGHFormatContract.ValidationError.invalid("invalid JANGH mapped bank shape") }
            let dtype: DType
            switch header.dtype {
            case "U32": dtype = .uint32
            case "F16": dtype = .float16
            case "F32": dtype = .float32
            case "I32": dtype = .int32
            default: throw JANGHFormatContract.ValidationError.invalid("invalid JANGH mapped bank dtype")
            }
            let result = try ResidentSafetensorsReader.loadRegion(
                file: file, offset: UInt64(location.fileOffset), length: location.byteCount,
                shape: header.shape, dtype: dtype)
            try JANGHBankLayout.requireReadyRowContiguous(result, role: location.tensor)
            return result
        }
        var mapped: [String: Projection] = [:]
        for (module, locations) in plan.projections where modules == nil || modules!.contains(module) {
            mapped[module] = try Projection(packed: read(locations.packed), scales: read(locations.scales))
        }
        var trellisMapped: [String: JANGTProjection] = [:]
        if modules == nil {
            for (module, t) in plan.trellis {
                func plain(_ location: JANGHTensorIndexPlan.Location) throws -> MLXArray {
                    try read(location)
                }
                trellisMapped[module] = try JANGTProjection(
                    k: source.contract.trellis[module]!, packed: plain(t.packed), scale: plain(t.scale),
                    signs: plain(t.signs), outlierColumnIndices: plain(t.outlierColumns),
                    outlierWeights: plain(t.outlierWeights), columnScales: t.columnScales.map(plain))
            }
        }
        // Reads used the leased descriptors; validate the complete bank set
        // before publication. The resulting arrays no longer depend on the files.
        for (name, file) in files {
            guard try JANGHHeaderAdapter.identity(file) == metadata.identities[name] else {
                throw JANGHFormatContract.ValidationError.invalid("JANGH shard changed while mapping")
            }
        }
        guard try JANGHHeaderAdapter.identity(index) == metadata.indexIdentity else {
            throw JANGHFormatContract.ValidationError.invalid("JANGH index changed while mapping")
        }
        self.contract = source.contract
        banks = mapped
        trellisBanks = trellisMapped
    }

    func trellisProjection(_ module: String) throws -> JANGTProjection {
        guard let value = trellisBanks[module] else {
            throw JANGHFormatContract.ValidationError.invalid("unknown mapped JANGT projection")
        }
        return value
    }

    func projection(_ module: String) throws -> Projection {
        guard let value = banks[module] else {
            throw JANGHFormatContract.ValidationError.invalid("unknown mapped JANGH projection")
        }
        return value
    }
}
