// Copyright © 2026 Osaurus AI
// SPDX-License-Identifier: MIT

import CryptoKit
import Foundation

/// The token stream represented by one diagnostic artifact.
///
/// Each stream has its own headerless UInt32 body.  The streams are kept
/// separate so a sampled lookahead cannot be mistaken for an accepted token.
public enum TokenIDTraceStreamKind: String, Codable, CaseIterable, Sendable {
    case prompt
    case sampled
    case iteratorReturned
    case batchYielded
    case adapterAccepted
    case sampledNotReturned
    case sampledNotYielded
}

/// The runtime path that owns the sampled-to-accepted relation.
public enum TokenIDTracePathKind: String, Codable, Sendable, Equatable {
    case direct
    case batch
}

/// The source-defined terminal rule for a token trace.
public enum TokenIDTraceStopRule: String, Codable, Sendable, Equatable {
    case eos
    case length
    case stopString
    case cancellation
    case error
}

/// Immutable source, binary, model, and tokenizer attestations carried by a
/// trace sidecar.  The values are supplied by the owning runtime; vMLX does
/// not invent them from a path or a process name.
public struct TokenIDTraceAttestations: Codable, Sendable, Equatable {
    public let model: String
    public let tokenizer: String
    public let binary: String
    public let source: String

    public init(
        model: String,
        tokenizer: String,
        binary: String,
        source: String
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.binary = binary
        self.source = source
    }
}

/// Count summary shared by every stream sidecar in one trace.
public struct TokenIDTraceLineageCounts: Codable, Sendable, Equatable {
    public let prompt: Int
    public let sampled: Int
    public let iteratorReturned: Int
    public let batchYielded: Int
    public let adapterAccepted: Int
    public let sampledNotReturned: Int
    public let sampledNotYielded: Int
    public let textEmitted: Int

    public init(
        prompt: Int = 0,
        sampled: Int = 0,
        iteratorReturned: Int = 0,
        batchYielded: Int = 0,
        adapterAccepted: Int = 0,
        sampledNotReturned: Int = 0,
        sampledNotYielded: Int = 0,
        textEmitted: Int = 0
    ) {
        self.prompt = prompt
        self.sampled = sampled
        self.iteratorReturned = iteratorReturned
        self.batchYielded = batchYielded
        self.adapterAccepted = adapterAccepted
        self.sampledNotReturned = sampledNotReturned
        self.sampledNotYielded = sampledNotYielded
        self.textEmitted = textEmitted
    }

    fileprivate func count(for stream: TokenIDTraceStreamKind) -> Int {
        switch stream {
        case .prompt: prompt
        case .sampled: sampled
        case .iteratorReturned: iteratorReturned
        case .batchYielded: batchYielded
        case .adapterAccepted: adapterAccepted
        case .sampledNotReturned: sampledNotReturned
        case .sampledNotYielded: sampledNotYielded
        }
    }
}

/// An ID and its ordinal in one trace stream.
public struct TokenIDTraceToken: Codable, Sendable, Equatable {
    public let id: Int
    public let ordinal: Int

    public init(id: Int, ordinal: Int) {
        self.id = id
        self.ordinal = ordinal
    }
}

/// Errors raised by the diagnostic encoder or its lineage validator.
public enum TokenIDTraceError: Error, LocalizedError, Sendable, Equatable {
    case alreadyStarted
    case alreadyFinished
    case invalidConfiguration(String)
    case unsupportedProposalPath(String)
    case invalidUInt32(stream: TokenIDTraceStreamKind, id: Int)
    case ordinalGap(stream: TokenIDTraceStreamKind, expected: Int, actual: Int)
    case countMismatch(String)
    case unclassifiedTerminalDelta(stream: TokenIDTraceStreamKind, delta: Int)
    case terminalRuleMismatch(expected: TokenIDTraceStopRule, actual: TokenIDTraceStopRule)
    case adapterAcceptanceMismatch(ordinal: Int)
    case adapterParticipationDisabled
    case sourceNotTerminated
    case adapterFinalizationRequired
    case sourceAlreadyTerminated
    case textIsNotTokenAuthority
    case writeFailed(String)

    public var errorDescription: String? {
        switch self {
        case .alreadyStarted:
            "token trace has already started"
        case .alreadyFinished:
            "token trace has already finished"
        case .invalidConfiguration(let detail):
            "invalid token trace configuration: \(detail)"
        case .unsupportedProposalPath(let path):
            "token trace requires MTP/speculation/proposals to be off; received \(path)"
        case .invalidUInt32(let stream, let id):
            "token trace ID \(id) in \(stream.rawValue) is not a UInt32"
        case .ordinalGap(let stream, let expected, let actual):
            "token trace ordinal gap in \(stream.rawValue): expected \(expected), got \(actual)"
        case .countMismatch(let detail):
            "token trace count/lineage mismatch: \(detail)"
        case .unclassifiedTerminalDelta(let stream, let delta):
            "token trace has unclassified terminal delta \(delta) in \(stream.rawValue)"
        case .terminalRuleMismatch(let expected, let actual):
            "token trace terminal rule changed from \(expected.rawValue) to \(actual.rawValue)"
        case .adapterAcceptanceMismatch(let ordinal):
            "token trace adapter acceptance does not match source ordinal \(ordinal)"
        case .adapterParticipationDisabled:
            "token trace has no external adapter participation"
        case .sourceNotTerminated:
            "token trace source has not terminated"
        case .adapterFinalizationRequired:
            "token trace requires external adapter finalization"
        case .sourceAlreadyTerminated:
            "token trace source events cannot change after source termination"
        case .textIsNotTokenAuthority:
            "decoded text cannot be used as token-ID authority"
        case .writeFailed(let detail):
            "token trace artifact write failed: \(detail)"
        }
    }
}

/// One encoded body and its deterministic JSON sidecar.
public struct TokenIDTraceEncodedArtifact: Sendable, Equatable {
    public let streamKind: TokenIDTraceStreamKind
    public let body: Data
    public let sidecar: Data

    public init(
        streamKind: TokenIDTraceStreamKind,
        body: Data,
        sidecar: Data
    ) {
        self.streamKind = streamKind
        self.body = body
        self.sidecar = sidecar
    }

    /// Write `*.ids` and `*.json` next to the caller-selected stem.
    ///
    /// The stem is not included in either artifact, so the sidecar remains
    /// byte-for-byte deterministic for the same lineage and attestations.
    public func write(to stem: URL) throws {
        let suffix = streamKind.rawValue
        do {
            try body.write(
                to: stem.appendingPathExtension("\(suffix).ids"),
                options: .atomic)
            try sidecar.write(
                to: stem.appendingPathExtension("\(suffix).json"),
                options: .atomic)
        } catch {
            throw TokenIDTraceError.writeFailed(error.localizedDescription)
        }
    }
}

/// The one shared encoder for Osaurus and vMLX token-evidence artifacts.
///
/// The body contains no header and stores each ID as four little-endian
/// bytes.  All labels, counts, ordinals, and attestations live in the
/// separate deterministic JSON sidecar.
public enum TokenIDTraceEncoder {
    public static let schemaVersion = 1

    private struct Sidecar: Codable {
        let schemaVersion: Int
        let streamKind: String
        let count: Int
        let sha256: String
        let ordinalStart: Int?
        let ordinalEnd: Int?
        let sourceOrdinalStart: Int?
        let sourceOrdinalEnd: Int?
        let stopRule: String
        let stopString: String?
        let stopStringRangeUTF8: [Int]?
        let stopStringSourceOrdinal: Int?
        let lineage: TokenIDTraceLineageCounts
        let attestations: TokenIDTraceAttestations
        let settingsHash: String
        let errorDetail: String?
    }

    /// Encode one stream.  `tokens` must have contiguous local ordinals
    /// starting at zero.  A lookahead stream keeps its source ordinal in the
    /// sidecar while its own body still has a contiguous one-item ordinal.
    public static func encode(
        streamKind: TokenIDTraceStreamKind,
        tokens: [TokenIDTraceToken],
        stopRule: TokenIDTraceStopRule,
        lineage: TokenIDTraceLineageCounts,
        attestations: TokenIDTraceAttestations,
        settingsHash: String,
        sourceOrdinalBounds: (start: Int, end: Int)? = nil,
        stopString: String? = nil,
        stopStringRangeUTF8: [Int]? = nil,
        stopStringSourceOrdinal: Int? = nil,
        errorDetail: String? = nil
    ) throws -> TokenIDTraceEncodedArtifact {
        try validateAttestations(attestations)
        guard isSHA256Hex(settingsHash) else {
            throw TokenIDTraceError.invalidConfiguration(
                "settingsHash must be a lowercase 64-character SHA-256")
        }
        guard lineage.count(for: streamKind) == tokens.count else {
            throw TokenIDTraceError.countMismatch(
                "\(streamKind.rawValue) sidecar count does not match lineage")
        }
        let hasStopMetadata =
            stopString != nil
            || stopStringRangeUTF8 != nil
            || stopStringSourceOrdinal != nil
        if stopRule == .stopString {
            guard let stopString, !stopString.isEmpty,
                stopStringRangeUTF8 != nil,
                stopStringSourceOrdinal != nil
            else {
                throw TokenIDTraceError.invalidConfiguration(
                    "stop-string artifacts require string, range, and source ordinal")
            }
        } else if hasStopMetadata {
            throw TokenIDTraceError.invalidConfiguration(
                "stop-string metadata requires the stopString terminal rule")
        }

        var expectedOrdinal = 0
        var body = Data(capacity: tokens.count * MemoryLayout<UInt32>.size)
        for token in tokens {
            guard token.ordinal == expectedOrdinal else {
                throw TokenIDTraceError.ordinalGap(
                    stream: streamKind,
                    expected: expectedOrdinal,
                    actual: token.ordinal)
            }
            guard token.id >= 0, token.id <= Int(UInt32.max) else {
                throw TokenIDTraceError.invalidUInt32(
                    stream: streamKind,
                    id: token.id)
            }
            var value = UInt32(token.id).littleEndian
            withUnsafeBytes(of: &value) { rawBytes in
                body.append(contentsOf: rawBytes)
            }
            expectedOrdinal += 1
        }

        if let sourceOrdinalBounds {
            guard sourceOrdinalBounds.start >= 0,
                sourceOrdinalBounds.end >= sourceOrdinalBounds.start
            else {
                throw TokenIDTraceError.ordinalGap(
                    stream: streamKind,
                    expected: sourceOrdinalBounds.start,
                    actual: sourceOrdinalBounds.end)
            }
            guard sourceOrdinalBounds.end - sourceOrdinalBounds.start + 1 == tokens.count else {
                throw TokenIDTraceError.countMismatch(
                    "source ordinal bounds do not cover the encoded stream")
            }
        }
        if let stopStringRangeUTF8 {
            guard stopStringRangeUTF8.count == 2,
                stopStringRangeUTF8[0] >= 0,
                stopStringRangeUTF8[1] > stopStringRangeUTF8[0]
            else {
                throw TokenIDTraceError.invalidConfiguration(
                    "stopStringRangeUTF8 must be a non-empty [start, end] range")
            }
        }
        if let stopStringSourceOrdinal, stopStringSourceOrdinal < 0 {
            throw TokenIDTraceError.invalidConfiguration(
                "stopStringSourceOrdinal must be non-negative")
        }

        let digest = SHA256.hash(data: body)
        let sha256 = digest.map { String(format: "%02x", $0) }.joined()
        let ordinalStart = tokens.first?.ordinal
        let ordinalEnd = tokens.last?.ordinal
        let sidecarValue = Sidecar(
            schemaVersion: schemaVersion,
            streamKind: streamKind.rawValue,
            count: tokens.count,
            sha256: sha256,
            ordinalStart: ordinalStart,
            ordinalEnd: ordinalEnd,
            sourceOrdinalStart: sourceOrdinalBounds?.start,
            sourceOrdinalEnd: sourceOrdinalBounds?.end,
            stopRule: stopRule.rawValue,
            stopString: stopString,
            stopStringRangeUTF8: stopStringRangeUTF8,
            stopStringSourceOrdinal: stopStringSourceOrdinal,
            lineage: lineage,
            attestations: attestations,
            settingsHash: settingsHash,
            errorDetail: errorDetail)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let sidecar = try encoder.encode(sidecarValue)
        return TokenIDTraceEncodedArtifact(
            streamKind: streamKind,
            body: body,
            sidecar: sidecar)
    }

    private static func validateAttestations(
        _ attestations: TokenIDTraceAttestations
    ) throws {
        let values = [
            ("model", attestations.model),
            ("tokenizer", attestations.tokenizer),
            ("binary", attestations.binary),
            ("source", attestations.source),
        ]
        guard values.allSatisfy({ !$0.1.isEmpty }) else {
            let missing = values.filter { $0.1.isEmpty }.map(\.0).joined(separator: ",")
            throw TokenIDTraceError.invalidConfiguration(
                "missing attestation(s): \(missing)")
        }
    }

    fileprivate static func validateConfiguration(
        attestations: TokenIDTraceAttestations,
        settingsHash: String
    ) throws {
        try validateAttestations(attestations)
        guard isSHA256Hex(settingsHash) else {
            throw TokenIDTraceError.invalidConfiguration(
                "settingsHash must be a lowercase 64-character SHA-256")
        }
    }

    private static func isSHA256Hex(_ value: String) -> Bool {
        value.count == 64
            && value.unicodeScalars.allSatisfy {
                (48 ... 57).contains($0.value) || (97 ... 102).contains($0.value)
            }
    }
}

/// Find the earliest text stop match for metadata only.  This helper never
/// supplies an ID and therefore cannot become token identity authority.
func tokenIDTraceFirstStopMatch(
    in text: String,
    stops: [String]
) -> (stop: String, rangeUTF8: [Int])? {
    var earliest: (stop: String, range: Range<String.Index>)?
    for stop in stops {
        guard let range = text.range(of: stop) else { continue }
        if let current = earliest, range.lowerBound >= current.range.lowerBound {
            continue
        }
        earliest = (stop, range)
    }
    guard let earliest else { return nil }
    let start = text[..<earliest.range.lowerBound].utf8.count
    let end = text[..<earliest.range.upperBound].utf8.count
    return (earliest.stop, [start, end])
}

/// Thread-safe public contract for a downstream adapter that has actually
/// accepted an ID.  vMLX exposes this contract but does not claim that an
/// external Osaurus adapter called it.
public final class TokenIDTraceAdapterContract: @unchecked Sendable {
    private let session: TokenIDTraceSession

    fileprivate init(session: TokenIDTraceSession) {
        self.session = session
    }

    /// Record one accepted ID at the adapter boundary.
    public func accept(tokenID: Int, ordinal: Int) throws {
        try session.recordAdapterAccepted(tokenID: tokenID, ordinal: ordinal)
    }

    /// The source terminal rule becomes available before this adapter phase
    /// can finalize. A nil value means the source is still running.
    public var sourceTerminalRule: TokenIDTraceStopRule? {
        session.sourceTerminalRule
    }

    /// Seal and, when configured, write the trace after source termination and
    /// all explicit adapter acceptance events have been recorded.
    public func finalizeAdapter() throws -> [TokenIDTraceEncodedArtifact] {
        try session.finalizeAdapter()
    }
}

/// Opt-in configuration for one generation request's token trace.
///
/// Construct a fresh configuration per request.  A nil
/// ``GenerateParameters/tokenIDTrace`` keeps the runtime on its existing
/// allocation, sampling, and output path and produces no artifacts.
public final class TokenIDTraceConfiguration: @unchecked Sendable {
    public let attestations: TokenIDTraceAttestations
    public let settingsHash: String
    public let outputStem: URL?
    public let externalAdapterParticipation: Bool

    let session: TokenIDTraceSession

    public init(
        attestations: TokenIDTraceAttestations,
        settingsHash: String,
        outputStem: URL? = nil,
        externalAdapterParticipation: Bool = false
    ) {
        self.attestations = attestations
        self.settingsHash = settingsHash
        self.outputStem = outputStem
        self.externalAdapterParticipation = externalAdapterParticipation
        self.session = TokenIDTraceSession(
            attestations: attestations,
            settingsHash: settingsHash,
            outputStem: outputStem,
            externalAdapterParticipation: externalAdapterParticipation)
    }

    /// The downstream acceptance surface.  It is deliberately separate from
    /// text output and from the sampled/lookahead streams.
    public var adapterContract: TokenIDTraceAdapterContract {
        session.adapterContract
    }

    /// The source terminal rule, if the runtime has reached its terminal
    /// event. External adapters use this as the phase boundary for accepts.
    public var sourceTerminalRule: TokenIDTraceStopRule? {
        session.sourceTerminalRule
    }

    /// Finalize an explicitly participating external adapter and return the
    /// sealed artifacts.
    public func finalizeAdapter() throws -> [TokenIDTraceEncodedArtifact] {
        try session.finalizeAdapter()
    }

    /// Return the finalized artifacts after the generation path has stopped.
    public func artifacts() throws -> [TokenIDTraceEncodedArtifact] {
        try session.artifacts()
    }

    internal func begin(
        pathKind: TokenIDTracePathKind,
        promptTokenIds: [Int],
        parameters: GenerateParameters,
        includeStopToken: Bool = false
    ) throws -> TokenIDTraceSession {
        try TokenIDTraceEncoder.validateConfiguration(
            attestations: attestations,
            settingsHash: settingsHash)
        guard !includeStopToken else {
            throw TokenIDTraceError.invalidConfiguration(
                "includeStopToken must remain false for this matrix")
        }
        if let strategy = parameters.draftStrategy,
            strategy.kindName != "none"
        {
            throw TokenIDTraceError.unsupportedProposalPath(strategy.kindName)
        }
        try session.begin(pathKind: pathKind, promptTokenIds: promptTokenIds)
        return session
    }

    /// Validate a diagnostic start before a caller selects a runtime path.
    /// This is used by dispatchers that can otherwise route into a proposal
    /// engine before a ``TokenIterator`` exists.
    internal func validateStart(
        parameters: GenerateParameters,
        includeStopToken: Bool = false
    ) throws {
        try TokenIDTraceEncoder.validateConfiguration(
            attestations: attestations,
            settingsHash: settingsHash)
        guard !includeStopToken else {
            throw TokenIDTraceError.invalidConfiguration(
                "includeStopToken must remain false for this matrix")
        }
        if let strategy = parameters.draftStrategy,
            strategy.kindName != "none"
        {
            throw TokenIDTraceError.unsupportedProposalPath(strategy.kindName)
        }
    }

    /// Explicit draft-model entry points use this before constructing a
    /// speculative iterator.  The caller supplies no replacement settings;
    /// the diagnostic path simply rejects the unsupported proposal path.
    internal func rejectProposalPath(_ path: String) throws {
        try TokenIDTraceEncoder.validateConfiguration(
            attestations: attestations,
            settingsHash: settingsHash)
        throw TokenIDTraceError.unsupportedProposalPath(path)
    }
}

/// Mutable, thread-safe lineage state shared by the runtime and the public
/// adapter contract.
public final class TokenIDTraceSession: @unchecked Sendable {
    private let lock = NSLock()
    private let attestations: TokenIDTraceAttestations
    private let settingsHash: String
    private let outputStem: URL?
    private let externalAdapterParticipation: Bool

    private var pathKind: TokenIDTracePathKind?
    private var prompt = [Int]()
    private var sampled = [Int]()
    private var iteratorReturned = [Int]()
    private var batchYielded = [Int]()
    private var adapterAccepted = [Int]()
    private var textEmittedCount = 0
    private var sourceTerminalRuleValue: TokenIDTraceStopRule?
    private var stopString: String?
    private var stopStringRangeUTF8: [Int]?
    private var stopStringSourceOrdinal: Int?
    private var errorDetail: String?
    private var failure: TokenIDTraceError?
    private var finalizedArtifacts: [TokenIDTraceEncodedArtifact]?

    init(
        attestations: TokenIDTraceAttestations,
        settingsHash: String,
        outputStem: URL?,
        externalAdapterParticipation: Bool
    ) {
        self.attestations = attestations
        self.settingsHash = settingsHash
        self.outputStem = outputStem
        self.externalAdapterParticipation = externalAdapterParticipation
    }

    var adapterContract: TokenIDTraceAdapterContract {
        TokenIDTraceAdapterContract(session: self)
    }

    var emitsExternalAdapterEvents: Bool {
        externalAdapterParticipation
    }

    var sourceTerminalRule: TokenIDTraceStopRule? {
        lock.lock()
        defer { lock.unlock() }
        return sourceTerminalRuleValue
    }

    func begin(
        pathKind: TokenIDTracePathKind,
        promptTokenIds: [Int]
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        if let failure {
            throw failure
        }
        if sourceTerminalRuleValue != nil || finalizedArtifacts != nil {
            throw TokenIDTraceError.alreadyFinished
        }
        if let existing = self.pathKind {
            guard existing == pathKind, self.prompt == promptTokenIds else {
                throw TokenIDTraceError.alreadyStarted
            }
            return
        }
        guard validateIDs(promptTokenIds, stream: .prompt) else {
            throw failure ?? TokenIDTraceError.invalidConfiguration("invalid prompt IDs")
        }
        self.pathKind = pathKind
        self.prompt = promptTokenIds
    }

    func replacePrompt(_ promptTokenIds: [Int]) {
        lock.lock()
        defer { lock.unlock() }
        guard failure == nil, finalizedArtifacts == nil else { return }
        guard sourceTerminalRuleValue == nil else {
            failure = .sourceAlreadyTerminated
            return
        }
        guard sampled.isEmpty, iteratorReturned.isEmpty, batchYielded.isEmpty else {
            failure = .countMismatch("effective prompt changed after sampling began")
            return
        }
        guard validateIDs(promptTokenIds, stream: .prompt) else { return }
        prompt = promptTokenIds
    }

    func recordSampled(_ tokenID: Int) {
        append(tokenID, to: .sampled)
    }

    func recordIteratorReturned(_ tokenID: Int) {
        append(tokenID, to: .iteratorReturned)
    }

    func recordBatchYielded(_ tokenID: Int) {
        append(tokenID, to: .batchYielded)
    }

    func recordTextEmitted() {
        lock.lock()
        if failure == nil, finalizedArtifacts == nil {
            textEmittedCount += 1
        }
        lock.unlock()
    }

    func recordStopString(
        stop: String,
        rangeUTF8: [Int]
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard failure == nil, finalizedArtifacts == nil else { return }
        guard sourceTerminalRuleValue == nil else {
            failure = .sourceAlreadyTerminated
            return
        }
        guard !stop.isEmpty else {
            failure = .invalidConfiguration("stop-string match must not be empty")
            return
        }
        guard stopString == nil else {
            failure = .countMismatch("multiple stop-string terminal matches")
            return
        }
        guard rangeUTF8.count == 2,
            rangeUTF8[0] >= 0,
            rangeUTF8[1] > rangeUTF8[0]
        else {
            failure = .invalidConfiguration("invalid stop-string match range")
            return
        }
        guard let pathKind else {
            failure = .invalidConfiguration("stop-string match before trace start")
            return
        }
        let sourceCount: Int
        switch pathKind {
        case .direct:
            sourceCount = iteratorReturned.count
        case .batch:
            sourceCount = batchYielded.count
        }
        guard sourceCount > 0 else {
            failure = .invalidConfiguration(
                "stop-string match has no observed source token")
            return
        }
        stopString = stop
        stopStringRangeUTF8 = rangeUTF8
        stopStringSourceOrdinal = sourceCount - 1
    }

    func recordRuntimeError(_ detail: String) {
        lock.lock()
        if failure == nil, finalizedArtifacts == nil {
            errorDetail = detail
        }
        lock.unlock()
    }

    /// Construction can fail after a model has entered prompt preparation but
    /// before the iterator is usable.  Preserve the input prompt while
    /// clearing all generated-event streams; no sampled or accepted event is
    /// inferred from a failed construction attempt.
    func discardRuntimeEventsForConstructionFailure() {
        lock.lock()
        guard failure == nil, finalizedArtifacts == nil else {
            lock.unlock()
            return
        }
        guard sourceTerminalRuleValue == nil else {
            failure = .sourceAlreadyTerminated
            lock.unlock()
            return
        }
        sampled.removeAll(keepingCapacity: false)
        iteratorReturned.removeAll(keepingCapacity: false)
        batchYielded.removeAll(keepingCapacity: false)
        adapterAccepted.removeAll(keepingCapacity: false)
        textEmittedCount = 0
        stopString = nil
        stopStringRangeUTF8 = nil
        stopStringSourceOrdinal = nil
        lock.unlock()
    }

    func recordAdapterAccepted(tokenID: Int, ordinal: Int) throws {
        lock.lock()
        defer { lock.unlock() }
        guard failure == nil else { throw failure! }
        guard finalizedArtifacts == nil else { throw TokenIDTraceError.alreadyFinished }
        guard externalAdapterParticipation else {
            let error = TokenIDTraceError.adapterParticipationDisabled
            failure = error
            throw error
        }
        guard let pathKind else {
            let error = TokenIDTraceError.invalidConfiguration(
                "adapter accepted before trace start")
            failure = error
            throw error
        }
        guard tokenID >= 0, tokenID <= Int(UInt32.max) else {
            let error = TokenIDTraceError.invalidUInt32(
                stream: .adapterAccepted,
                id: tokenID)
            failure = error
            throw error
        }
        let expectedOrdinal = adapterAccepted.count
        guard ordinal == expectedOrdinal else {
            let error = TokenIDTraceError.ordinalGap(
                stream: .adapterAccepted,
                expected: expectedOrdinal,
                actual: ordinal)
            failure = error
            throw error
        }
        let source = pathKind == .direct ? iteratorReturned : batchYielded
        guard ordinal < source.count, source[ordinal] == tokenID else {
            let error = TokenIDTraceError.adapterAcceptanceMismatch(ordinal: ordinal)
            failure = error
            throw error
        }
        adapterAccepted.append(tokenID)
    }

    /// Finalize a trace, validate all source-specific terminal deltas, and
    /// optionally write the deterministic artifacts at the configured stem.
    /// With external adapter participation this records only the source
    /// terminal; the adapter must call `finalizeAdapter()` later.
    func finish(
        stopRule: TokenIDTraceStopRule,
        errorDetail: String? = nil
    ) throws -> [TokenIDTraceEncodedArtifact] {
        lock.lock()
        defer { lock.unlock() }
        if let failure {
            throw failure
        }
        if let finalizedArtifacts {
            let existingRule = sourceTerminalRuleValue ?? .error
            if existingRule != stopRule {
                throw TokenIDTraceError.terminalRuleMismatch(
                    expected: existingRule,
                    actual: stopRule)
            }
            return finalizedArtifacts
        }
        if let existingRule = sourceTerminalRuleValue {
            guard existingRule == stopRule else {
                throw TokenIDTraceError.terminalRuleMismatch(
                    expected: existingRule,
                    actual: stopRule)
            }
            return []
        }
        guard let pathKind else {
            throw TokenIDTraceError.invalidConfiguration("trace did not start")
        }
        do {
            try validateTerminalLocked(
                pathKind: pathKind,
                stopRule: stopRule)
            sourceTerminalRuleValue = stopRule
            if let errorDetail {
                self.errorDetail = errorDetail
            }
            guard !externalAdapterParticipation else { return [] }
            return try finalizeLocked(stopRule: stopRule)
        } catch {
            let traceError = asTraceError(error)
            failure = traceError
            throw traceError
        }
    }

    /// Seal and write an explicitly participating external adapter trace.
    func finalizeAdapter() throws -> [TokenIDTraceEncodedArtifact] {
        lock.lock()
        defer { lock.unlock() }
        if let failure {
            throw failure
        }
        guard externalAdapterParticipation else {
            let error = TokenIDTraceError.adapterParticipationDisabled
            failure = error
            throw error
        }
        if let finalizedArtifacts {
            return finalizedArtifacts
        }
        guard let sourceTerminalRuleValue else {
            let error = TokenIDTraceError.sourceNotTerminated
            failure = error
            throw error
        }
        guard let pathKind else {
            let error = TokenIDTraceError.invalidConfiguration("trace did not start")
            failure = error
            throw error
        }
        do {
            try validateTerminalLocked(
                pathKind: pathKind,
                stopRule: sourceTerminalRuleValue)
            return try finalizeLocked(stopRule: sourceTerminalRuleValue)
        } catch {
            let traceError = asTraceError(error)
            failure = traceError
            throw traceError
        }
    }

    func artifacts() throws -> [TokenIDTraceEncodedArtifact] {
        lock.lock()
        defer { lock.unlock() }
        if let failure {
            throw failure
        }
        guard let finalizedArtifacts else {
            if externalAdapterParticipation, sourceTerminalRuleValue != nil {
                throw TokenIDTraceError.adapterFinalizationRequired
            }
            throw TokenIDTraceError.invalidConfiguration(
                "artifacts requested before terminal trace")
        }
        return finalizedArtifacts
    }

    private func finalizeLocked(
        stopRule: TokenIDTraceStopRule
    ) throws -> [TokenIDTraceEncodedArtifact] {
        guard let pathKind else {
            throw TokenIDTraceError.invalidConfiguration("trace did not start")
        }
        let artifacts = try makeArtifactsLocked(
            pathKind: pathKind,
            stopRule: stopRule)
        if let outputStem {
            do {
                for artifact in artifacts {
                    try artifact.write(to: outputStem)
                }
            } catch {
                let traceError =
                    (error as? TokenIDTraceError)
                    ?? .writeFailed(error.localizedDescription)
                failure = traceError
                throw traceError
            }
        }
        finalizedArtifacts = artifacts
        return artifacts
    }

    private func makeArtifactsLocked(
        pathKind: TokenIDTracePathKind,
        stopRule: TokenIDTraceStopRule
    ) throws -> [TokenIDTraceEncodedArtifact] {
        let sampledNotReturned: Int
        let sampledNotYielded: Int
        let exposedKind: TokenIDTraceStreamKind
        let exposedIDs: [Int]
        let unacceptedKind: TokenIDTraceStreamKind
        switch pathKind {
        case .direct:
            sampledNotReturned = max(0, sampled.count - iteratorReturned.count)
            sampledNotYielded = 0
            exposedKind = .iteratorReturned
            exposedIDs = iteratorReturned
            unacceptedKind = .sampledNotReturned
        case .batch:
            sampledNotReturned = 0
            sampledNotYielded = max(0, sampled.count - batchYielded.count)
            exposedKind = .batchYielded
            exposedIDs = batchYielded
            unacceptedKind = .sampledNotYielded
        }
        let counts = TokenIDTraceLineageCounts(
            prompt: prompt.count,
            sampled: sampled.count,
            iteratorReturned: iteratorReturned.count,
            batchYielded: batchYielded.count,
            adapterAccepted: adapterAccepted.count,
            sampledNotReturned: sampledNotReturned,
            sampledNotYielded: sampledNotYielded,
            textEmitted: textEmittedCount)
        let unaccepted = Array(sampled.dropFirst(exposedIDs.count))
        let unacceptedSourceBounds: (Int, Int)?
        if unaccepted.isEmpty {
            unacceptedSourceBounds = nil
        } else {
            unacceptedSourceBounds = (exposedIDs.count, sampled.count - 1)
        }

        let streams: [(TokenIDTraceStreamKind, [Int], (Int, Int)?)] = [
            (.prompt, prompt, nil),
            (.sampled, sampled, nil),
            (exposedKind, exposedIDs, nil),
            (.adapterAccepted, adapterAccepted, nil),
            (unacceptedKind, unaccepted, unacceptedSourceBounds),
        ]
        return try streams.map { entry in
            let (kind, ids, sourceBounds) = entry
            let tokens = ids.enumerated().map {
                TokenIDTraceToken(id: $0.element, ordinal: $0.offset)
            }
            return try TokenIDTraceEncoder.encode(
                streamKind: kind,
                tokens: tokens,
                stopRule: stopRule,
                lineage: counts,
                attestations: attestations,
                settingsHash: settingsHash,
                sourceOrdinalBounds: sourceBounds,
                stopString: stopString,
                stopStringRangeUTF8: stopStringRangeUTF8,
                stopStringSourceOrdinal: stopStringSourceOrdinal,
                errorDetail: errorDetail)
        }
    }

    private func asTraceError(_ error: Error) -> TokenIDTraceError {
        if let error = error as? TokenIDTraceError {
            return error
        }
        return .invalidConfiguration(error.localizedDescription)
    }

    private func append(_ tokenID: Int, to stream: TokenIDTraceStreamKind) {
        lock.lock()
        defer { lock.unlock() }
        guard failure == nil, finalizedArtifacts == nil else { return }
        guard sourceTerminalRuleValue == nil else {
            failure = .sourceAlreadyTerminated
            return
        }
        guard validateID(tokenID, stream: stream) else { return }
        switch stream {
        case .sampled:
            sampled.append(tokenID)
        case .iteratorReturned:
            iteratorReturned.append(tokenID)
        case .batchYielded:
            batchYielded.append(tokenID)
        default:
            failure = .invalidConfiguration(
                "runtime cannot append to \(stream.rawValue) directly")
        }
    }

    private func validateID(
        _ tokenID: Int,
        stream: TokenIDTraceStreamKind
    ) -> Bool {
        guard tokenID >= 0, tokenID <= Int(UInt32.max) else {
            failure = .invalidUInt32(stream: stream, id: tokenID)
            return false
        }
        return true
    }

    private func validateIDs(
        _ ids: [Int],
        stream: TokenIDTraceStreamKind
    ) -> Bool {
        for id in ids where !validateID(id, stream: stream) {
            return false
        }
        return true
    }

    private func validateTerminalLocked(
        pathKind: TokenIDTracePathKind,
        stopRule: TokenIDTraceStopRule
    ) throws {
        guard stopString == nil || stopRule == .stopString else {
            throw TokenIDTraceError.countMismatch(
                "stop-string metadata does not match terminal rule")
        }
        if stopRule == .stopString, stopString == nil {
            throw TokenIDTraceError.countMismatch(
                "stop-string terminal rule has no recorded match")
        }
        let source: [Int]
        let inactiveSource: [Int]
        switch pathKind {
        case .direct:
            source = iteratorReturned
            inactiveSource = batchYielded
        case .batch:
            source = batchYielded
            inactiveSource = iteratorReturned
        }
        guard inactiveSource.isEmpty else {
            throw TokenIDTraceError.countMismatch(
                "inactive source stream is not empty for \(pathKind.rawValue) path")
        }
        guard source.count <= sampled.count else {
            throw TokenIDTraceError.unclassifiedTerminalDelta(
                stream: pathKind == .direct ? .sampledNotReturned : .sampledNotYielded,
                delta: sampled.count - source.count)
        }
        guard Array(sampled.prefix(source.count)) == source else {
            throw TokenIDTraceError.countMismatch(
                "active source stream is not the sampled prefix")
        }
        let sourceCount = source.count
        let delta = sampled.count - sourceCount
        guard delta >= 0, delta <= 1 else {
            throw TokenIDTraceError.unclassifiedTerminalDelta(
                stream: pathKind == .direct ? .sampledNotReturned : .sampledNotYielded,
                delta: delta)
        }

        switch pathKind {
        case .direct:
            switch stopRule {
            case .eos, .length, .stopString:
                guard delta == 1 else {
                    throw TokenIDTraceError.unclassifiedTerminalDelta(
                        stream: .sampledNotReturned,
                        delta: delta)
                }
            case .cancellation, .error:
                break
            }
        case .batch:
            if stopRule == .eos, delta != 1 {
                throw TokenIDTraceError.unclassifiedTerminalDelta(
                    stream: .sampledNotYielded,
                    delta: delta)
            }
            if stopRule == .length, delta != 0 {
                throw TokenIDTraceError.unclassifiedTerminalDelta(
                    stream: .sampledNotYielded,
                    delta: delta)
            }
        default:
            throw TokenIDTraceError.invalidConfiguration(
                "unsupported runtime path \(pathKind.rawValue)")
        }

        if stopRule == .stopString {
            guard let sourceOrdinal = stopStringSourceOrdinal else {
                throw TokenIDTraceError.invalidConfiguration(
                    "stop-string source ordinal is outside the active source stream")
            }
            guard sourceOrdinal >= 0, sourceOrdinal < source.count else {
                throw TokenIDTraceError.invalidConfiguration(
                    "stop-string source ordinal is outside the active source stream")
            }
        }
        let maxAdapterCount =
            pathKind == .direct && stopRule == .eos
            ? max(0, source.count - 1)
            : source.count
        guard adapterAccepted.count <= maxAdapterCount else {
            throw TokenIDTraceError.countMismatch(
                "adapterAccepted exceeds the accepted source prefix")
        }
        guard Array(source.prefix(adapterAccepted.count)) == adapterAccepted else {
            if let mismatch = adapterAccepted.enumerated().first(where: {
                $0.offset >= source.count || source[$0.offset] != $0.element
            }) {
                throw TokenIDTraceError.adapterAcceptanceMismatch(
                    ordinal: mismatch.offset)
            }
            throw TokenIDTraceError.adapterAcceptanceMismatch(
                ordinal: adapterAccepted.count)
        }
    }
}
