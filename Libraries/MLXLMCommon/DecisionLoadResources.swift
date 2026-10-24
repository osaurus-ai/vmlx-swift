import Foundation
import MLX

/// Resource ownership for a non-generative decision backbone. Uses the same resolved scheduling
/// limit, allocator-pool policy and per-model residency lease as the chat factory, without
/// constructing a ModelContext (which requires vocabulary-logit generation).
public final class DecisionLoadResources {
    private var lease: ModelResidency.Lease?
    private let priorAllocatorCap: Int?

    public init(directory: URL, configuration: LoadConfiguration) {
        let facts = LoadBundleFacts.inspect(bundleURL: directory)
        let workingSet = MLX.GPU.maxRecommendedWorkingSetBytes() ?? Int.max
        let limit = configuration.resolvedSchedulingMemoryLimit(facts: facts, recommendedWorkingSetBytes: workingSet)
        let pool = configuration.maxResidentBytes.applyAsCacheLimitInt(physicalMemory: facts.physicalMemory)
        let native = applyResidentPoolProcessMemoryLimitsIfNeeded(facts: facts, allocatorCacheLimit: pool)
        if native == nil, let pool {
            priorAllocatorCap = Memory.cacheLimit
            Memory.cacheLimit = pool
        } else { priorAllocatorCap = nil }
        if native == nil, let raw = limit.applyAsCacheLimitInt(physicalMemory: facts.physicalMemory) {
            Memory.memoryLimit = max(1024 * 1024 * 1024, min(raw, workingSet))
        }
        let wire = SystemWireBudget.snapshot()
        lease = ModelResidency.shared.acquire(.init(weightBytes: Int(clamping: facts.gpuResidentWeightBytes),
            enabled: configuration.keepModelInMemory, physicalMemory: Int(clamping: facts.physicalMemory),
            workingSet: workingSet, systemWireLimit: wire.userWireLimitBytes)).lease
    }

    /// Host restores this scalar after load, before transferring the lease to the model owner.
    public var previousAllocatorCacheLimit: Int? { priorAllocatorCap }
}
