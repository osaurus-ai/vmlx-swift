import Foundation

/// Validates batch shape before model work and resolves seeds once per request.
/// Sequential sampling still shares the prepared conditioning; it is the lowest
/// memory implementation of a native batch, not repeated prompt encoding.
public struct ImageBatchPlan: Sendable, Equatable {
    public let seeds: [UInt64]
    public let batchSize: Int

    public static func validate(_ request: ImageGenRequest, maximumImages: Int) throws {
        guard request.numImages >= 1, request.numImages <= maximumImages else {
            throw FluxError.invalidField(field: "numImages", value: "\(request.numImages)", allowed: "1...\(maximumImages)")
        }
        guard request.batchSize >= 1 else {
            throw FluxError.invalidField(field: "batchSize", value: "\(request.batchSize)", allowed: "a positive integer")
        }
        if let seeds = request.seeds, seeds.count != request.numImages {
            throw FluxError.invalidField(field: "seeds", value: "\(seeds.count) seeds", allowed: "exactly \(request.numImages) seeds")
        }
    }

    public init(_ request: ImageGenRequest, maximumImages: Int, maximumParallelImages: Int) throws {
        try Self.validate(request, maximumImages: maximumImages)
        precondition(maximumParallelImages > 0)
        batchSize = min(request.batchSize, request.numImages, maximumParallelImages)
        seeds = request.seeds ?? (0..<request.numImages).map { index in
            request.seed.map { $0 &+ UInt64(index) } ?? UInt64.random(in: 0...UInt64(UInt32.max))
        }
    }
}
