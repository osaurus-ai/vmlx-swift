import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

private let traceAttestations = TokenIDTraceAttestations(
    model: "model-attestation",
    tokenizer: "tokenizer-attestation",
    binary: "binary-attestation",
    source: "source-attestation")

private let traceSettingsHash = String(repeating: "a", count: 64)

private func traceConfiguration(
    externalAdapterParticipation: Bool = false,
    outputStem: URL? = nil
) -> TokenIDTraceConfiguration {
    TokenIDTraceConfiguration(
        attestations: traceAttestations,
        settingsHash: traceSettingsHash,
        outputStem: outputStem,
        externalAdapterParticipation: externalAdapterParticipation)
}

private func artifact(
    _ artifacts: [TokenIDTraceEncodedArtifact],
    kind: TokenIDTraceStreamKind
) -> TokenIDTraceEncodedArtifact {
    artifacts.first { $0.streamKind == kind }!
}

private func sidecarText(_ artifact: TokenIDTraceEncodedArtifact) -> String {
    String(data: artifact.sidecar, encoding: .utf8)!
}

private struct TokenIDTraceFixedPieceTokenizer: MLXLMCommon.Tokenizer {
    let pieces: [String]

    var vocabularySize: Int { pieces.count + 1 }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { pieces.indices.contains($0) ? pieces[$0] : "" }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { pieces.firstIndex(of: token) }

    func convertIdToToken(_ id: Int) -> String? {
        pieces.indices.contains(id) ? pieces[id] : nil
    }

    var bosToken: String? = nil
    var eosToken: String? = nil
    var unknownToken: String? = nil

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private struct TokenIDTracePositionPieceTokenizer: MLXLMCommon.Tokenizer {
    let pieces: [String]

    var vocabularySize: Int { 8 }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [0] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.indices.map { pieces[min($0, pieces.count - 1)] }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { pieces.firstIndex(of: token) }
    func convertIdToToken(_ id: Int) -> String? {
        pieces.indices.contains(id) ? pieces[id] : nil
    }

    var bosToken: String? = nil
    var eosToken: String? = nil
    var unknownToken: String? = nil

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [0] }
}

private final class TokenIDTraceBatchModel: Module, LanguageModel,
    KVCacheDimensionProvider, @unchecked Sendable
{
    let vocabularySize = 8
    var kvHeads: [Int] { [1] }

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let batch = inputs.shape.first ?? 1
        let length = inputs.shape.count > 1 ? inputs.shape[1] : inputs.size
        var values = [Float](
            repeating: -1,
            count: batch * length * vocabularySize)
        for row in 0 ..< batch * length {
            values[row * vocabularySize] = 0
        }
        return MLXArray(values).reshaped(batch, length, vocabularySize)
    }
}

private func tokenIDTraceBatchEngine(
    pieces: [String],
    maxBatchSize: Int = 2
) -> BatchEngine {
    let model = TokenIDTraceBatchModel()
    let tokenizer = TokenIDTracePositionPieceTokenizer(pieces: pieces)
    var configuration = ModelConfiguration(id: "token-trace-batch-terminal-ordering")
    configuration.eosTokenIds = []
    let processor = TestInputProcessor(
        tokenizer: tokenizer,
        configuration: configuration,
        messageGenerator: DefaultMessageGenerator())
    let context = ModelContext(
        configuration: configuration,
        model: model,
        processor: processor,
        tokenizer: tokenizer)
    return BatchEngine(context: context, maxBatchSize: maxBatchSize)
}

private let tokenIDTraceFlushPieces = [
    "one, ", "two, ", "three, ", "four, ", "five, ",
]

@Suite("BatchEngine token trace terminal ordering", .serialized)
struct BatchEngineTokenTraceTerminalOrderingTests {
    @Test("direct and batched high-level streams expose diagnostic IDs to an adapter")
    func highLevelStreamsExposeDiagnosticIDs() async throws {
        for maxBatchSize in [1, 2] {
            let engine = tokenIDTraceBatchEngine(
                pieces: tokenIDTraceFlushPieces,
                maxBatchSize: maxBatchSize)
            let configuration = traceConfiguration(externalAdapterParticipation: true)
            var parameters = GenerateParameters(
                maxTokens: tokenIDTraceFlushPieces.count,
                temperature: 0)
            parameters.tokenIDTrace = configuration

            let stream = await engine.generate(
                input: LMInput(tokens: MLXArray([Int32(0)])),
                parameters: parameters)
            var observed = [TokenIDTraceToken]()
            var sawTerminalInfo = false
            for await event in stream {
                switch event {
                case .tokenID(let id, let ordinal):
                    observed.append(TokenIDTraceToken(id: id, ordinal: ordinal))
                    try configuration.adapterContract.accept(tokenID: id, ordinal: ordinal)
                case .info:
                    sawTerminalInfo = true
                default:
                    break
                }
            }
            await engine.shutdown()

            #expect(sawTerminalInfo)
            #expect(observed.count == tokenIDTraceFlushPieces.count)
            #expect(observed.map(\.ordinal) == Array(tokenIDTraceFlushPieces.indices))
            #expect(observed.allSatisfy { $0.id == 0 })
            #expect(configuration.sourceTerminalRule == .length)
            let artifacts = try configuration.finalizeAdapter()
            let expectedSourceKind: TokenIDTraceStreamKind =
                maxBatchSize == 1 ? .iteratorReturned : .batchYielded
            let unexpectedSourceKind: TokenIDTraceStreamKind =
                maxBatchSize == 1 ? .batchYielded : .iteratorReturned
            #expect(artifacts.contains { $0.streamKind == expectedSourceKind })
            #expect(!artifacts.contains { $0.streamKind == unexpectedSourceKind })
            #expect(
                artifact(artifacts, kind: .adapterAccepted).body
                    == artifact(artifacts, kind: expectedSourceKind).body
            )
        }
    }

    @Test("trace-off high-level streams emit no diagnostic IDs")
    func traceOffStreamsEmitNoDiagnosticIDs() async {
        for maxBatchSize in [1, 2] {
            let engine = tokenIDTraceBatchEngine(
                pieces: tokenIDTraceFlushPieces,
                maxBatchSize: maxBatchSize)
            let stream = await engine.generate(
                input: LMInput(tokens: MLXArray([Int32(0)])),
                parameters: GenerateParameters(
                    maxTokens: tokenIDTraceFlushPieces.count,
                    temperature: 0))
            var diagnosticIDCount = 0
            for await event in stream {
                if case .tokenID = event { diagnosticIDCount += 1 }
            }
            await engine.shutdown()
            #expect(diagnosticIDCount == 0)
        }
    }

    @Test("deferred trace lets a flush-only stop override length")
    func deferredFlushOnlyStopOverridesLength() async throws {
        let engine = tokenIDTraceBatchEngine(pieces: tokenIDTraceFlushPieces)
        let configuration = traceConfiguration()
        var parameters = GenerateParameters(
            maxTokens: tokenIDTraceFlushPieces.count,
            temperature: 0,
            extraStopStrings: ["five"])
        parameters.tokenIDTrace = configuration

        let stream = await engine.generate(
            input: LMInput(tokens: MLXArray([Int32(0)])),
            parameters: parameters)
        var completion: GenerateCompletionInfo?
        for await event in stream {
            if case .info(let info) = event { completion = info }
        }
        await engine.shutdown()

        #expect(completion?.stopReason == .stop)
        let artifacts = try configuration.artifacts()
        let sampledSidecar = sidecarText(artifact(artifacts, kind: .sampled))
        #expect(sampledSidecar.contains(#""stopRule":"stopString""#))
        #expect(sampledSidecar.contains(#""stopString":"five""#))
    }

    @Test("external adapter trace remains open through a flush-only stop")
    func externalAdapterFlushOnlyStopFinalizes() async throws {
        let engine = tokenIDTraceBatchEngine(pieces: tokenIDTraceFlushPieces)
        let configuration = traceConfiguration(externalAdapterParticipation: true)
        var parameters = GenerateParameters(
            maxTokens: tokenIDTraceFlushPieces.count,
            temperature: 0,
            extraStopStrings: ["five"])
        parameters.tokenIDTrace = configuration

        let stream = await engine.generate(
            input: LMInput(tokens: MLXArray([Int32(0)])),
            parameters: parameters)
        for await _ in stream {}
        await engine.shutdown()

        #expect(configuration.sourceTerminalRule == .stopString)
        for ordinal in tokenIDTraceFlushPieces.indices {
            try configuration.adapterContract.accept(tokenID: 0, ordinal: ordinal)
        }
        let artifacts = try configuration.finalizeAdapter()
        let sampledSidecar = sidecarText(artifact(artifacts, kind: .sampled))
        #expect(sampledSidecar.contains(#""stopRule":"stopString""#))
        #expect(sampledSidecar.contains(#""stopString":"five""#))
    }
}

@Test("shared encoder writes deterministic little-endian UInt32 bodies")
func tokenIDTraceEncoderIsDeterministic() throws {
    let tokens = [
        TokenIDTraceToken(id: 1, ordinal: 0),
        TokenIDTraceToken(id: 256, ordinal: 1),
        TokenIDTraceToken(id: Int(UInt32.max), ordinal: 2),
    ]
    let lineage = TokenIDTraceLineageCounts(sampled: tokens.count)

    let first = try TokenIDTraceEncoder.encode(
        streamKind: .sampled,
        tokens: tokens,
        stopRule: .length,
        lineage: lineage,
        attestations: traceAttestations,
        settingsHash: traceSettingsHash)
    let second = try TokenIDTraceEncoder.encode(
        streamKind: .sampled,
        tokens: tokens,
        stopRule: .length,
        lineage: lineage,
        attestations: traceAttestations,
        settingsHash: traceSettingsHash)

    #expect(first.body == second.body)
    #expect(first.sidecar == second.sidecar)
    #expect(
        Array(first.body) == [
            1, 0, 0, 0,
            0, 1, 0, 0,
            255, 255, 255, 255,
        ])
    #expect(sidecarText(first).contains("\"schemaVersion\":1"))
    #expect(sidecarText(first).contains("\"streamKind\":\"sampled\""))
    #expect(
        sidecarText(first).contains(
            "\"settingsHash\":\"" + traceSettingsHash + "\""))
    let digest = SHA256.hash(data: first.body)
        .map { String(format: "%02x", $0) }
        .joined()
    #expect(sidecarText(first).contains("\"sha256\":\"" + digest + "\""))
}

@Test("encoder rejects invalid IDs, ordinal gaps, and lineage count mismatch")
func tokenIDTraceEncoderFailsClosed() throws {
    let oneSample = TokenIDTraceLineageCounts(sampled: 1)

    #expect(throws: TokenIDTraceError.self) {
        try TokenIDTraceEncoder.encode(
            streamKind: .sampled,
            tokens: [TokenIDTraceToken(id: -1, ordinal: 0)],
            stopRule: .error,
            lineage: oneSample,
            attestations: traceAttestations,
            settingsHash: traceSettingsHash)
    }
    #expect(throws: TokenIDTraceError.self) {
        try TokenIDTraceEncoder.encode(
            streamKind: .sampled,
            tokens: [TokenIDTraceToken(id: 3, ordinal: 1)],
            stopRule: .error,
            lineage: oneSample,
            attestations: traceAttestations,
            settingsHash: traceSettingsHash)
    }
    #expect(throws: TokenIDTraceError.self) {
        try TokenIDTraceEncoder.encode(
            streamKind: .sampled,
            tokens: [TokenIDTraceToken(id: 3, ordinal: 0)],
            stopRule: .error,
            lineage: TokenIDTraceLineageCounts(sampled: 0),
            attestations: traceAttestations,
            settingsHash: traceSettingsHash)
    }
    #expect(throws: TokenIDTraceError.self) {
        try TokenIDTraceEncoder.encode(
            streamKind: .sampled,
            tokens: [],
            stopRule: .length,
            lineage: TokenIDTraceLineageCounts(),
            attestations: traceAttestations,
            settingsHash: traceSettingsHash,
            stopString: "stop")
    }
}

@Test("direct length keeps one sampled-but-not-returned ID")
func directLengthLineage() throws {
    let configuration = traceConfiguration()
    let session = try configuration.begin(
        pathKind: .direct,
        promptTokenIds: [10, 11],
        parameters: GenerateParameters())
    session.recordSampled(20)
    session.recordSampled(21)
    session.recordIteratorReturned(20)

    let artifacts = try session.finish(stopRule: .length)
    #expect(
        Array(artifact(artifacts, kind: .sampledNotReturned).body) == [
            21, 0, 0, 0,
        ])
    #expect(
        sidecarText(artifact(artifacts, kind: .sampled)).contains(
            "\"sampled\":2"))
    #expect(
        sidecarText(artifact(artifacts, kind: .iteratorReturned)).contains(
            "\"ordinalStart\":0"))
}

@Test("direct EOS excludes the returned stop ID from adapter acceptance")
func directEOSLineage() throws {
    let configuration = traceConfiguration(externalAdapterParticipation: true)
    let session = try configuration.begin(
        pathKind: .direct,
        promptTokenIds: [10],
        parameters: GenerateParameters())
    session.recordSampled(20)
    session.recordSampled(21)
    session.recordSampled(22)
    session.recordIteratorReturned(20)
    session.recordIteratorReturned(21)
    try configuration.adapterContract.accept(tokenID: 20, ordinal: 0)
    #expect(try session.finish(stopRule: .eos).isEmpty)
    let artifacts = try configuration.adapterContract.finalizeAdapter()
    #expect(
        Array(artifact(artifacts, kind: .adapterAccepted).body) == [
            20, 0, 0, 0,
        ])

    let rejectedConfiguration = traceConfiguration(externalAdapterParticipation: true)
    let rejectedSession = try rejectedConfiguration.begin(
        pathKind: .direct,
        promptTokenIds: [10],
        parameters: GenerateParameters())
    rejectedSession.recordSampled(20)
    rejectedSession.recordSampled(21)
    rejectedSession.recordSampled(22)
    rejectedSession.recordIteratorReturned(20)
    rejectedSession.recordIteratorReturned(21)
    try rejectedConfiguration.adapterContract.accept(tokenID: 20, ordinal: 0)
    try rejectedConfiguration.adapterContract.accept(tokenID: 21, ordinal: 1)
    #expect(throws: TokenIDTraceError.self) {
        try rejectedSession.finish(stopRule: .eos)
    }
}

@Test("direct stop string records metadata without using text as ID authority")
func directStopStringLineage() throws {
    let configuration = traceConfiguration()
    let session = try configuration.begin(
        pathKind: .direct,
        promptTokenIds: [1],
        parameters: GenerateParameters())
    session.recordSampled(30)
    session.recordSampled(31)
    session.recordIteratorReturned(30)
    session.recordStopString(stop: "stop", rangeUTF8: [4, 8])

    let artifacts = try session.finish(stopRule: .stopString)
    let sidecar = sidecarText(artifact(artifacts, kind: .iteratorReturned))
    #expect(sidecar.contains(#""stopStringSourceOrdinal":0"#))
    #expect(sidecar.contains("\"stopRule\":\"stopString\""))
    #expect(sidecar.contains("\"stopString\":\"stop\""))
    #expect(sidecar.contains("\"stopStringRangeUTF8\":[4,8]"))
    #expect(
        Array(artifact(artifacts, kind: .sampledNotReturned).body) == [
            31, 0, 0, 0,
        ])

    let invalidRangeConfiguration = traceConfiguration()
    let invalidRange = try invalidRangeConfiguration.begin(
        pathKind: .direct,
        promptTokenIds: [1],
        parameters: GenerateParameters())
    invalidRange.recordSampled(32)
    invalidRange.recordSampled(33)
    invalidRange.recordIteratorReturned(32)
    invalidRange.recordStopString(stop: "stop", rangeUTF8: [-1, 3])
    #expect(throws: TokenIDTraceError.self) {
        try invalidRange.finish(stopRule: .stopString)
    }
}

@Test("same-lineage validation rejects a wrong prefix and inactive source")
func sameLineageValidationFailsClosed() throws {
    let wrongPrefixConfiguration = traceConfiguration()
    let wrongPrefix = try wrongPrefixConfiguration.begin(
        pathKind: .direct,
        promptTokenIds: [1],
        parameters: GenerateParameters())
    wrongPrefix.recordSampled(1)
    wrongPrefix.recordSampled(2)
    wrongPrefix.recordIteratorReturned(9)
    #expect(throws: TokenIDTraceError.self) {
        try wrongPrefix.finish(stopRule: .length)
    }

    let inactiveConfiguration = traceConfiguration()
    let inactive = try inactiveConfiguration.begin(
        pathKind: .direct,
        promptTokenIds: [1],
        parameters: GenerateParameters())
    inactive.recordSampled(3)
    inactive.recordSampled(4)
    inactive.recordIteratorReturned(3)
    inactive.recordBatchYielded(4)
    #expect(throws: TokenIDTraceError.self) {
        try inactive.finish(stopRule: .length)
    }
}

@Test("batch lineage rejects a non-prefix exposed stream")
func batchSameLineageValidationFailsClosed() throws {
    let configuration = traceConfiguration()
    let session = try configuration.begin(
        pathKind: .batch,
        promptTokenIds: [1],
        parameters: GenerateParameters())
    session.recordSampled(5)
    session.recordSampled(6)
    session.recordBatchYielded(9)
    #expect(throws: TokenIDTraceError.self) {
        try session.finish(stopRule: .eos)
    }
}

@Test("batch EOS and length keep batch-specific terminal semantics")
func batchTerminalLineage() throws {
    let eosConfiguration = traceConfiguration()
    let eos = try eosConfiguration.begin(
        pathKind: .batch,
        promptTokenIds: [1],
        parameters: GenerateParameters())
    eos.recordSampled(40)
    let eosArtifacts = try eos.finish(stopRule: .eos)
    #expect(
        Array(artifact(eosArtifacts, kind: .sampledNotYielded).body) == [
            40, 0, 0, 0,
        ])

    let lengthConfiguration = traceConfiguration()
    let length = try lengthConfiguration.begin(
        pathKind: .batch,
        promptTokenIds: [1],
        parameters: GenerateParameters())
    length.recordSampled(41)
    length.recordBatchYielded(41)
    let lengthArtifacts = try length.finish(stopRule: .length)
    #expect(artifact(lengthArtifacts, kind: .sampledNotYielded).body.isEmpty)
    #expect(
        sidecarText(artifact(lengthArtifacts, kind: .batchYielded)).contains(
            "\"batchYielded\":1"))
}

@Test("cancellation and construction error do not synthesize accepted IDs")
func cancellationAndConstructionErrorLineage() throws {
    let cancelledConfiguration = traceConfiguration()
    let cancelled = try cancelledConfiguration.begin(
        pathKind: .direct,
        promptTokenIds: [2],
        parameters: GenerateParameters())
    cancelled.recordSampled(50)
    cancelled.recordSampled(51)
    cancelled.recordIteratorReturned(50)
    let cancelledArtifacts = try cancelled.finish(stopRule: .cancellation)
    #expect(
        Array(artifact(cancelledArtifacts, kind: .sampledNotReturned).body) == [
            51, 0, 0, 0,
        ])
    #expect(artifact(cancelledArtifacts, kind: .adapterAccepted).body.isEmpty)

    let constructionConfiguration = traceConfiguration()
    let construction = try constructionConfiguration.begin(
        pathKind: .direct,
        promptTokenIds: [3],
        parameters: GenerateParameters())
    construction.recordSampled(60)
    construction.discardRuntimeEventsForConstructionFailure()
    construction.recordRuntimeError("injected construction error")
    let constructionArtifacts = try construction.finish(stopRule: .error)
    #expect(artifact(constructionArtifacts, kind: .sampled).body.isEmpty)
    #expect(artifact(constructionArtifacts, kind: .iteratorReturned).body.isEmpty)
    #expect(
        sidecarText(artifact(constructionArtifacts, kind: .sampled)).contains(
            "injected construction error"))
}

@Test("adapter contract requires matching source ordinals")
func adapterContractLineage() throws {
    let invalidConfiguration = traceConfiguration(externalAdapterParticipation: true)
    let invalidSession = try invalidConfiguration.begin(
        pathKind: .direct,
        promptTokenIds: [4],
        parameters: GenerateParameters())
    invalidSession.recordSampled(70)
    invalidSession.recordSampled(71)
    invalidSession.recordIteratorReturned(70)
    try invalidSession.finish(stopRule: .length)
    #expect(throws: TokenIDTraceError.self) {
        try invalidConfiguration.adapterContract.accept(tokenID: 999, ordinal: 0)
    }

    let wrongOrdinalConfiguration = traceConfiguration(
        externalAdapterParticipation: true)
    let wrongOrdinalSession = try wrongOrdinalConfiguration.begin(
        pathKind: .direct,
        promptTokenIds: [4],
        parameters: GenerateParameters())
    wrongOrdinalSession.recordSampled(70)
    wrongOrdinalSession.recordSampled(71)
    wrongOrdinalSession.recordIteratorReturned(70)
    try wrongOrdinalSession.finish(stopRule: .length)
    #expect(throws: TokenIDTraceError.self) {
        try wrongOrdinalConfiguration.adapterContract.accept(tokenID: 70, ordinal: 2)
    }

    let configuration = traceConfiguration(externalAdapterParticipation: true)
    let session = try configuration.begin(
        pathKind: .direct,
        promptTokenIds: [4],
        parameters: GenerateParameters())
    session.recordSampled(70)
    session.recordSampled(71)
    session.recordIteratorReturned(70)
    try session.finish(stopRule: .length)
    try configuration.adapterContract.accept(tokenID: 70, ordinal: 0)
    let artifacts = try configuration.adapterContract.finalizeAdapter()
    #expect(
        Array(artifact(artifacts, kind: .adapterAccepted).body) == [
            70, 0, 0, 0,
        ])
}

@Test("terminal count mismatch poisons a trace")
func terminalCountMismatchFailsClosed() throws {
    let configuration = traceConfiguration()
    let session = try configuration.begin(
        pathKind: .direct,
        promptTokenIds: [4],
        parameters: GenerateParameters())
    session.recordSampled(72)
    session.recordSampled(73)
    session.recordSampled(74)
    session.recordIteratorReturned(72)
    #expect(throws: TokenIDTraceError.self) {
        try session.finish(stopRule: .length)
    }
}

@Test("external adapter finalization is a separate phase after source termination")
func externalAdapterTwoPhaseFinalization() throws {
    let blockedConfiguration = traceConfiguration(externalAdapterParticipation: true)
    let blockedSession = try blockedConfiguration.begin(
        pathKind: .direct,
        promptTokenIds: [8],
        parameters: GenerateParameters())
    blockedSession.recordSampled(79)
    blockedSession.recordIteratorReturned(79)
    try blockedConfiguration.adapterContract.accept(tokenID: 79, ordinal: 0)
    #expect(throws: TokenIDTraceError.self) {
        try blockedConfiguration.adapterContract.finalizeAdapter()
    }

    let configuration = traceConfiguration(externalAdapterParticipation: true)
    let session = try configuration.begin(
        pathKind: .direct,
        promptTokenIds: [8],
        parameters: GenerateParameters())
    session.recordSampled(80)
    session.recordIteratorReturned(80)
    try configuration.adapterContract.accept(tokenID: 80, ordinal: 0)
    #expect(configuration.sourceTerminalRule == nil)

    session.recordSampled(81)
    session.recordSampled(82)
    session.recordIteratorReturned(81)
    #expect(try session.finish(stopRule: .length).isEmpty)
    #expect(configuration.sourceTerminalRule == .length)
    try configuration.adapterContract.accept(tokenID: 81, ordinal: 1)
    let artifacts = try configuration.adapterContract.finalizeAdapter()
    #expect(!artifacts.isEmpty)
    #expect(
        Array(artifact(artifacts, kind: .adapterAccepted).body) == [
            80, 0, 0, 0,
            81, 0, 0, 0,
        ])
    #expect(try configuration.artifacts() == artifacts)
}

@Test("premature adapter finalization and post-terminal source events fail closed")
func externalAdapterPhaseOrderingFailsClosed() throws {
    let prematureConfiguration = traceConfiguration(
        externalAdapterParticipation: true)
    let prematureSession = try prematureConfiguration.begin(
        pathKind: .direct,
        promptTokenIds: [9],
        parameters: GenerateParameters())
    #expect(throws: TokenIDTraceError.self) {
        try prematureConfiguration.adapterContract.finalizeAdapter()
    }
    prematureSession.recordSampled(90)
    prematureSession.recordSampled(91)
    prematureSession.recordIteratorReturned(90)
    #expect(throws: TokenIDTraceError.self) {
        try prematureSession.finish(stopRule: .length)
    }

    let lateEventConfiguration = traceConfiguration(
        externalAdapterParticipation: true)
    let lateEventSession = try lateEventConfiguration.begin(
        pathKind: .direct,
        promptTokenIds: [9],
        parameters: GenerateParameters())
    lateEventSession.recordSampled(92)
    lateEventSession.recordSampled(93)
    lateEventSession.recordIteratorReturned(92)
    try lateEventSession.finish(stopRule: .length)
    lateEventSession.recordSampled(94)
    #expect(throws: TokenIDTraceError.self) {
        try lateEventConfiguration.adapterContract.finalizeAdapter()
    }
}

@Test("artifact write failure remains observable and blocks retry")
func artifactWriteFailureFailsClosed() throws {
    let configuration = traceConfiguration(
        outputStem: URL(fileURLWithPath: "/dev/null/vmlx-token-trace"))
    let session = try configuration.begin(
        pathKind: .direct,
        promptTokenIds: [10],
        parameters: GenerateParameters())
    session.recordSampled(100)
    session.recordSampled(101)
    session.recordIteratorReturned(100)
    #expect(throws: TokenIDTraceError.self) {
        try session.finish(stopRule: .length)
    }

    var surfacedWriteFailure = false
    do {
        _ = try configuration.artifacts()
    } catch let error as TokenIDTraceError {
        if case .writeFailed = error {
            surfacedWriteFailure = true
        }
    }
    #expect(surfacedWriteFailure)
    #expect(throws: TokenIDTraceError.self) {
        try session.finish(stopRule: .length)
    }
}

@Test("tracing remains opt-in and defaults to no external adapter")
func tracingDefaultsAreDisabled() {
    #expect(GenerateParameters().tokenIDTrace == nil)
    #expect(traceConfiguration().externalAdapterParticipation == false)
    #expect(throws: TokenIDTraceError.self) {
        try traceConfiguration().adapterContract.accept(tokenID: 1, ordinal: 0)
    }
}

@Test("diagnostic start rejects MTP and proposal paths without changing settings")
func diagnosticStartRejectsProposalPaths() throws {
    let configuration = traceConfiguration()
    var parameters = GenerateParameters()
    parameters.draftStrategy = .dflash(
        drafterPath: URL(fileURLWithPath: "/tmp/not-loaded"),
        blockSize: 4)
    #expect(throws: TokenIDTraceError.self) {
        try configuration.begin(
            pathKind: .direct,
            promptTokenIds: [5],
            parameters: parameters)
    }
    #expect(parameters.draftStrategy?.kindName == "dflash")

    var mtpParameters = GenerateParameters()
    mtpParameters.draftStrategy = .nativeMTP(depth: 1)
    #expect(throws: TokenIDTraceError.self) {
        try traceConfiguration().begin(
            pathKind: .direct,
            promptTokenIds: [5],
            parameters: mtpParameters)
    }
    #expect(GenerateParameters().tokenIDTrace == nil)
}

@Test("batch cancellation before sampling has no generated-event counts")
func batchCancellationBeforeSampling() throws {
    let configuration = traceConfiguration()
    let session = try configuration.begin(
        pathKind: .batch,
        promptTokenIds: [6],
        parameters: GenerateParameters())
    let artifacts = try session.finish(stopRule: .cancellation)
    #expect(artifact(artifacts, kind: .sampled).body.isEmpty)
    #expect(artifact(artifacts, kind: .batchYielded).body.isEmpty)
    #expect(artifact(artifacts, kind: .sampledNotYielded).body.isEmpty)
}

@Test("batch error after sampling before yield keeps one unyielded ID")
func batchErrorAfterSamplingBeforeYield() throws {
    let configuration = traceConfiguration()
    let session = try configuration.begin(
        pathKind: .batch,
        promptTokenIds: [7],
        parameters: GenerateParameters())
    session.recordSampled(80)
    session.recordRuntimeError("injected batch error before yield")
    let artifacts = try session.finish(stopRule: .error)
    #expect(
        Array(artifact(artifacts, kind: .sampledNotYielded).body) == [
            80, 0, 0, 0,
        ])
    #expect(
        sidecarText(artifact(artifacts, kind: .sampled)).contains(
            "injected batch error before yield"))
}

@Test("consumer termination cannot reclassify a flush-only stop string")
func downstreamTerminationPreservesTraceCancellation() throws {
    let pieces = [
        "one, ", "two, ", "three, ", "four, ", "five, ",
        "six, ", "seven, ", "eight, ", "nine, ", "ten.",
    ]

    func drive(
        stopStrings: [String],
        tokenIDs: [Int],
        terminateConsumer: Bool
    ) throws -> (
        rule: TokenIDTraceStopRule,
        artifacts: [TokenIDTraceEncodedArtifact],
        legacyRule: TokenIDTraceStopRule,
        stopReasonKind: String,
        hitBeforeFlush: Bool,
        hitAfterFlush: Bool,
        stopMatchRecorded: Bool,
        consumerTerminated: Bool,
        onTokenReturnedFalse: Bool,
        stoppedAtToken: Int?,
        terminatedDuringOnToken: Bool,
        terminatedDuringGenerationEnd: Bool,
        firstNonemptyEmissionToken: Int,
        firstNonemptyChunk: String,
        iteratorReturned: Int,
        sampled: Int
    ) {
        func legacyPostFlushOnlyStopRule(
            stopReason: GenerateStopReason,
            stopSequenceHitAfterGenerationEnd: Bool
        ) -> TokenIDTraceStopRule {
            // Revert comparator: a post-flush match wins before the
            // cancellation reason is considered.
            if stopSequenceHitAfterGenerationEnd {
                return .stopString
            }
            switch stopReason {
            case .stop:
                return .eos
            case .length:
                return .length
            case .cancelled:
                return .cancellation
            }
        }

        func stopReasonKind(_ reason: GenerateStopReason) -> String {
            switch reason {
            case .stop:
                return "stop"
            case .length:
                return "length"
            case .cancelled:
                return "cancelled"
            }
        }

        let configuration = traceConfiguration()
        let session = try configuration.begin(
            pathKind: .direct,
            promptTokenIds: [100],
            parameters: GenerateParameters())
        var handler = TextToolTokenLoopHandler(
            tokenizer: TokenIDTraceFixedPieceTokenizer(pieces: pieces),
            format: .json,
            stopStringMatcher: StopStringMatcher(stopStrings: stopStrings),
            tokenTrace: session)

        var consumerTerminated = false
        var activeToken: Int? = nil
        var generationEndStarted = false
        var terminatedDuringOnToken = false
        var terminatedDuringGenerationEnd = false
        var firstNonemptyEmissionToken = -1
        var firstNonemptyChunk = ""
        let emit: (sending Generation) -> AsyncStream<Generation>.Continuation.YieldResult = {
            event in
            if case .chunk(let text) = event {
                if !text.isEmpty {
                    if firstNonemptyEmissionToken == -1 {
                        firstNonemptyEmissionToken = activeToken ?? -1
                        firstNonemptyChunk = text
                    }
                    if terminateConsumer && !consumerTerminated {
                        consumerTerminated = true
                        if generationEndStarted {
                            terminatedDuringGenerationEnd = true
                        } else {
                            terminatedDuringOnToken = true
                        }
                        return .terminated
                    }
                }
            }
            return .enqueued(remaining: .max)
        }

        var stopReason: GenerateStopReason = .length
        var sampled = 0
        var iteratorReturned = 0
        var onTokenReturnedFalse = false
        var stoppedAtToken: Int? = nil
        for token in tokenIDs {
            session.recordSampled(token)
            sampled += 1
            session.recordIteratorReturned(token)
            iteratorReturned += 1
            activeToken = token
            let keepGoing = handler.onToken(token, emit: emit)
            activeToken = nil
            if !keepGoing {
                onTokenReturnedFalse = true
                stoppedAtToken = token
                stopReason = handler.stopSequenceHit ? .stop : .cancelled
                break
            }
        }
        // The direct iterator samples one lookahead beyond its returned prefix.
        session.recordSampled(999)
        sampled += 1

        let hitBeforeFlush = handler.stopSequenceHit
        generationEndStarted = true
        handler.onGenerationEnd(emit: emit)
        let hitAfterFlush = handler.stopSequenceHit
        let rule = tokenIDTraceStopRule(
            stopReason: stopReason,
            stopSequenceHitBeforeGenerationEnd: hitBeforeFlush,
            stopSequenceHitAfterGenerationEnd: hitAfterFlush)
        let legacyRule = legacyPostFlushOnlyStopRule(
            stopReason: stopReason,
            stopSequenceHitAfterGenerationEnd: hitAfterFlush)

        if rule == .stopString {
            guard let match = handler.stopStringMatch else {
                throw TokenIDTraceError.invalidConfiguration(
                    "test stop match was not retained")
            }
            session.recordStopString(stop: match.stop, rangeUTF8: match.rangeUTF8)
        }
        let artifacts = try session.finish(stopRule: rule)
        return (
            rule,
            artifacts,
            legacyRule,
            stopReasonKind(stopReason),
            hitBeforeFlush,
            hitAfterFlush,
            handler.stopStringMatch != nil,
            consumerTerminated,
            onTokenReturnedFalse,
            stoppedAtToken,
            terminatedDuringOnToken,
            terminatedDuringGenerationEnd,
            firstNonemptyEmissionToken,
            firstNonemptyChunk,
            iteratorReturned,
            sampled)
    }

    // Intervention arm: five tokens leave "five" in the detokenizer's held
    // tail. The first nonempty emitted chunk is "on" from onToken(token: 4),
    // so downstream termination makes the source reason cancellation. The
    // held stop is discovered only by onGenerationEnd.
    let cancelled = try drive(
        stopStrings: ["five"],
        tokenIDs: Array(0 ..< 5),
        terminateConsumer: true)
    #expect(cancelled.consumerTerminated)
    #expect(cancelled.onTokenReturnedFalse)
    #expect(cancelled.stoppedAtToken == 4)
    #expect(cancelled.terminatedDuringOnToken)
    #expect(!cancelled.terminatedDuringGenerationEnd)
    #expect(cancelled.firstNonemptyEmissionToken == 4)
    #expect(cancelled.firstNonemptyChunk == "on")
    #expect(cancelled.iteratorReturned == 5)
    #expect(cancelled.sampled == 6)
    #expect(cancelled.stopReasonKind == "cancelled")
    #expect(!cancelled.hitBeforeFlush)
    #expect(cancelled.hitAfterFlush)
    #expect(cancelled.stopMatchRecorded)
    #expect(cancelled.legacyRule == .stopString)
    #expect(cancelled.rule == .cancellation)
    let cancelledSidecar = sidecarText(artifact(cancelled.artifacts, kind: .sampled))
    #expect(cancelledSidecar.contains(#""stopRule":"cancellation""#))
    #expect(!cancelledSidecar.contains(#""stopString":"five""#))

    // Control arm: the exact same five-token prefix and stop, with the
    // consumer left open, produce a valid flush-time stop-string trace.
    let flushed = try drive(
        stopStrings: ["five"],
        tokenIDs: Array(0 ..< 5),
        terminateConsumer: false)
    #expect(!flushed.consumerTerminated)
    #expect(!flushed.onTokenReturnedFalse)
    #expect(flushed.firstNonemptyEmissionToken == 4)
    #expect(flushed.firstNonemptyChunk == "on")
    #expect(flushed.iteratorReturned == 5)
    #expect(flushed.sampled == 6)
    #expect(flushed.stopReasonKind == "length")
    #expect(!flushed.hitBeforeFlush)
    #expect(flushed.hitAfterFlush)
    #expect(flushed.stopMatchRecorded)
    #expect(flushed.rule == .stopString)
    let flushedSidecar = sidecarText(artifact(flushed.artifacts, kind: .sampled))
    #expect(flushedSidecar.contains(#""stopRule":"stopString""#))
    #expect(flushedSidecar.contains(#""stopString":"five""#))
    #expect(flushedSidecar.contains(#""stopStringRangeUTF8":[23,27]"#))

    // A genuine match before end-of-generation remains a stop-string trace.
    let matched = try drive(
        stopStrings: ["three"],
        tokenIDs: Array(pieces.indices),
        terminateConsumer: false)
    #expect(matched.onTokenReturnedFalse)
    #expect(matched.stoppedAtToken == 6)
    #expect(matched.stopReasonKind == "stop")
    #expect(!matched.consumerTerminated)
    #expect(!matched.terminatedDuringOnToken)
    #expect(!matched.terminatedDuringGenerationEnd)
    #expect(matched.hitBeforeFlush)
    #expect(matched.hitAfterFlush)
    #expect(matched.stopMatchRecorded)
    #expect(matched.rule == .stopString)
}
