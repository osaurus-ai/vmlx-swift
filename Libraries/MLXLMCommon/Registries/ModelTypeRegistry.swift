// Copyright © 2024 Apple Inc.

import Foundation

public actor ModelTypeRegistry {

    /// Creates an empty registry.
    public init() {
        self.creators = [:]
    }

    /// Creates a registry with given creators.
    public init(creators: [String: (Data) throws -> any LanguageModel]) {
        self.creators = creators
    }

    private var creators: [String: (Data) throws -> any LanguageModel]

    /// Add a new model to the type registry.
    public func registerModelType(
        _ type: String, creator: @escaping (Data) throws -> any LanguageModel
    ) {
        creators[type] = creator
    }

    /// Every `model_type` this registry can instantiate.
    ///
    /// Exposed so the two registries can be checked against each other. A family with both a
    /// text-only and a multimodal implementation is registered in two different factories, in files
    /// that share no symbol — and comparing the key sets is the only way to notice one was missed.
    /// See `DualPathFamilies`.
    public var registeredModelTypes: Set<String> { Set(creators.keys) }

    /// Given a `modelType` and configuration data instantiate a new `LanguageModel`.
    public func createModel(configuration: Data, modelType: String) throws -> sending LanguageModel
    {
        guard let creator = creators[modelType] else {
            throw ModelFactoryError.unsupportedModelType(modelType)
        }
        return try creator(configuration)
    }

}
