// Copyright © 2024 Apple Inc.

import Foundation

/// The one closure shape a registry stores.
///
/// Every model is built through this and only this. A family that has nothing to select simply
/// ignores the request; a family with optional towers reads it. There is deliberately no second,
/// modality-aware dictionary alongside this one — a parallel path is how the request came to be
/// silently dropped before, with the modality-aware initialisers present but unreachable.
public typealias ModelCreator =
    (Data, Set<ModelRuntimeRequestModality>?) throws -> any LanguageModel

public actor ModelTypeRegistry {

    /// Creates an empty registry.
    public init() {
        self.creators = [:]
    }

    /// Creates a registry with given creators.
    public init(creators: [String: ModelCreator]) {
        self.creators = creators
    }

    private var creators: [String: ModelCreator]

    /// Add a new model to the type registry.
    public func registerModelType(_ type: String, creator: @escaping ModelCreator) {
        creators[type] = creator
    }

    /// Register a model that has nothing to select — it is built the same way whatever is asked of
    /// it. Adapts to the single stored shape; it does not add a second dispatch path.
    public func registerModelType(
        _ type: String, creator: @escaping (Data) throws -> any LanguageModel
    ) {
        creators[type] = { data, _ in try creator(data) }
    }

    /// Every `model_type` this registry can instantiate.
    ///
    /// Exposed so the two registries can be checked against each other. A family with both a
    /// text-only and a multimodal implementation is registered in two different factories, in files
    /// that share no symbol — and comparing the key sets is the only way to notice one was missed.
    /// See `DualPathFamilies`.
    public var registeredModelTypes: Set<String> { Set(creators.keys) }

    /// Given a `modelType` and configuration data instantiate a new `LanguageModel`.
    ///
    /// - Parameter requesting: which lanes the caller intends to use, or `nil` for "everything this
    ///   configuration supports". Families that build optional towers consult it; the rest ignore it.
    public func createModel(
        configuration: Data, modelType: String,
        requesting: Set<ModelRuntimeRequestModality>? = nil
    ) throws -> sending LanguageModel {
        guard let creator = creators[modelType] else {
            throw ModelFactoryError.unsupportedModelType(modelType)
        }
        return try creator(configuration, requesting)
    }

}
