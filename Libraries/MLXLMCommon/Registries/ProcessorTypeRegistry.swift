// Copyright © 2024 Apple Inc.

import Foundation

private typealias ProcessorCreator = (Data, any Tokenizer) throws -> any UserInputProcessor

/// The same table serves actor-isolated construction and synchronous admission.
/// Callers never invoke a creator while holding the lock.
private final class ProcessorCreators: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [String: ProcessorCreator]
    private var revision: UInt64 = 0

    init(_ entries: [String: ProcessorCreator]) { self.entries = entries }

    func contains(_ type: String) -> Bool {
        lock.withLock { entries[type] != nil }
    }

    func register(_ type: String, creator: @escaping ProcessorCreator) {
        lock.withLock {
            entries[type] = creator
            revision &+= 1
        }
    }

    var registrationVersion: UInt64 { lock.withLock { revision } }

    func creator(for type: String) -> ProcessorCreator? {
        lock.withLock { entries[type] }
    }
}

public actor ProcessorTypeRegistry {

    /// Creates an empty registry.
    public init() {
        self.creators = ProcessorCreators([:])
    }

    /// Creates a registry with given creators.
    public init(creators: [String: (Data, any Tokenizer) throws -> any UserInputProcessor]) {
        self.creators = ProcessorCreators(creators)
    }

    private nonisolated let creators: ProcessorCreators

    /// Read registration without creating a processor, tokenizer or model.
    /// Registrations become visible before registerProcessorType returns.
    public nonisolated func containsProcessorType(_ type: String) -> Bool {
        creators.contains(type)
    }

    /// Lets admission caches discard a verdict made before a registration changed.
    public nonisolated var registrationVersion: UInt64 { creators.registrationVersion }

    /// Add a new model to the type registry.
    public func registerProcessorType(
        _ type: String,
        creator:
            @escaping (
                Data,
                any Tokenizer
            ) throws -> any UserInputProcessor
    ) {
        creators.register(type, creator: creator)
    }

    /// Given a `processorType` and configuration data instantiate a new `UserInputProcessor`.
    public func createModel(configuration: Data, processorType: String, tokenizer: any Tokenizer)
        throws -> sending any UserInputProcessor
    {
        guard let creator = creators.creator(for: processorType) else {
            throw ModelFactoryError.unsupportedProcessorType(processorType)
        }
        return try creator(configuration, tokenizer)
    }

}
