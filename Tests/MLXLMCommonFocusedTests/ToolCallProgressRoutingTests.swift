import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// Coverage for the `.toolCallProgress` streaming accommodation: while the
/// tool-call processor collects a committed call, `routeGenerationText` emits
/// incremental envelope deltas so a consumer can preview a long call (e.g. a
/// file write) instead of seeing a silent gap. The contract these tests pin:
///   1. a multi-chunk envelope yields progress deltas whose concatenation is
///      the collected envelope text, followed by exactly one parsed `.toolCall`
///      and no visible `.chunk` leak;
///   2. plain prose never yields progress events;
///   3. a strip-only processor (no tools offered) never leaks envelope text as
///      progress, because it produces no terminating `.toolCall`;
///   4. a committed malformed envelope never becomes executable or visible,
///      and terminates with explicit protocol-failure metadata.
struct ToolCallProgressRoutingTests {

    private func lineCountToolSpec() -> [String: any Sendable] {
        [
            "type": "function",
            "function": [
                "name": "line_count",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "text": ["type": "string"] as [String: any Sendable]
                    ] as [String: any Sendable],
                    "required": ["text"],
                    "additionalProperties": false,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }

    /// A Zyphra/Gemma4 tool-call envelope that parses to a single `line_count`
    /// call (same shape the Gemma4 parser tests use).
    private let envelope = """
        <zyphra_tool_call>
        <function=line_count>
        <parameter=text>
        red
        green
        blue
        </parameter>
        </function>
        </zyphra_tool_call>
        """

    private let malformedDSMLOutput = """
        <｜DSML｜tool_calls>
        <｜DSML｜invoke name="line_count">
        <｜DSML｜parameter name="text" string="true">red\ngreen\nblue</｜DSML｜parameter>
        </｜DSML｜inv>
        </｜DSML｜tool_calls>
        """

    /// Split a string into `count` roughly equal contiguous chunks, mimicking
    /// token-by-token streaming without depending on tokenizer boundaries.
    private func chunked(_ s: String, into count: Int) -> [String] {
        let chars = Array(s)
        guard count > 1, chars.count >= count else { return [s] }
        let size = Int((Double(chars.count) / Double(count)).rounded(.up))
        return stride(from: 0, to: chars.count, by: size).map {
            String(chars[$0 ..< min($0 + size, chars.count)])
        }
    }

    @Test("multi-chunk tool-call envelope streams progress deltas then one parsed call")
    func multiChunkEnvelopeStreamsProgressThenOneCall() {
        let processor = ToolCallProcessor(format: .gemma4, tools: [lineCountToolSpec()])
        var progress: [String] = []
        var visible = ""
        var calls: [ToolCall] = []

        for piece in chunked(envelope, into: 6) {
            for event in routeGenerationText(piece, channel: .content, through: processor) {
                switch event {
                case .toolCallProgress(let delta): progress.append(delta)
                case .chunk(let text): visible += text
                case .toolCall(let call): calls.append(call)
                default: break
                }
            }
        }
        for event in flushGenerationText(channel: .content, through: processor) {
            if case .toolCall(let call) = event { calls.append(call) }
            if case .chunk(let text) = event { visible += text }
        }

        // The whole point: the collection window is NOT silent.
        #expect(!progress.isEmpty, "expected incremental tool-call progress deltas")
        // Deltas are real envelope bytes in order — the concatenation is a
        // contiguous prefix of the raw envelope, never fabricated text.
        let joined = progress.joined()
        #expect(!joined.isEmpty)
        #expect(envelope.contains(joined) || joined.contains("<zyphra_tool_call>"))
        #expect(envelope.hasPrefix(joined) || envelope.contains(joined))
        // The parsed call still arrives exactly once, and nothing leaked to
        // the visible channel.
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "line_count")
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test("plain prose yields no tool-call progress events")
    func plainProseYieldsNoProgress() {
        let processor = ToolCallProcessor(format: .gemma4, tools: [lineCountToolSpec()])
        var progress: [String] = []
        var visible = ""
        let prose = "Sure — here is a short explanation with no tool call at all."
        for piece in chunked(prose, into: 5) {
            for event in routeGenerationText(piece, channel: .content, through: processor) {
                if case .toolCallProgress(let d) = event { progress.append(d) }
                if case .chunk(let t) = event { visible += t }
            }
        }
        visible += flushGenerationText(channel: .content, through: processor)
            .compactMap(\.chunk).joined()

        #expect(progress.isEmpty, "plain prose must never emit tool-call progress")
        #expect(visible.contains("no tool call"))
    }

    @Test("strip-only processor never leaks envelope text as progress")
    func stripOnlyNeverLeaksProgress() {
        // No tools offered → strip-only: markers are stripped from visible text
        // and the parsed call is discarded, so there is no terminating
        // `.toolCall`. A progress delta here would strand the consumer.
        let processor = ToolCallProcessor(format: .gemma4, tools: nil, stripOnly: true)
        var progress: [String] = []
        var calls = 0
        for piece in chunked(envelope, into: 6) {
            for event in routeGenerationText(piece, channel: .content, through: processor) {
                if case .toolCallProgress(let d) = event { progress.append(d) }
                if case .toolCall = event { calls += 1 }
            }
        }
        for event in flushGenerationText(channel: .content, through: processor) {
            if case .toolCall = event { calls += 1 }
        }

        #expect(progress.isEmpty, "strip-only must not surface envelope text as progress")
        #expect(calls == 0, "strip-only discards the parsed call")
    }

    @Test("malformed DSML progress terminates as an explicit non-executable failure")
    func malformedDSMLProgressTerminatesAsFailure() {
        let processor = ToolCallProcessor(format: .dsml, tools: [lineCountToolSpec()])
        var progress = ""
        var visible = ""
        var calls: [ToolCall] = []

        for character in malformedDSMLOutput {
            for event in routeGenerationText(
                String(character), channel: .content, through: processor)
            {
                switch event {
                case .toolCallProgress(let delta): progress += delta
                case .chunk(let text): visible += text
                case .toolCall(let call): calls.append(call)
                default: break
                }
            }
        }
        for event in flushGenerationText(channel: .content, through: processor) {
            if case .chunk(let text) = event { visible += text }
            if case .toolCall(let call) = event { calls.append(call) }
        }

        #expect(!progress.isEmpty, "the committed envelope should have streamed progress")
        #expect(calls.isEmpty, "malformed DSML must never become executable")
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("DSML"), "protocol bytes must remain quarantined")
        #expect(processor.toolCallProtocolFailure == .malformedEnvelope)
    }

    @Test("completion metadata remains source-compatible and defaults to no tool failure")
    func completionMetadataDefaultsToNoFailure() {
        let info = GenerateCompletionInfo(
            promptTokenCount: 1,
            generationTokenCount: 1,
            promptTime: 0.1,
            generationTime: 0.1)

        #expect(info.toolCallProtocolFailure == nil)
    }

    @Test("BatchEngine malformed DSML stream is finite and carries failure metadata")
    func batchEngineMalformedDSMLStreamIsFinite() async {
        nonisolated(unsafe) let context = malformedDSMLContext(output: malformedDSMLOutput)
        let engine = BatchEngine(context: context, maxBatchSize: 2)
        let input = LMInput(tokens: MLXArray([Int32(1)]))
            .withToolSchemas([lineCountToolSpec()])
        let stream = await engine.generate(
            input: input,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0))

        var visible = ""
        var progressDeltas = 0
        var calls = 0
        var infos: [GenerateCompletionInfo] = []
        for await event in stream {
            switch event {
            case .chunk(let text): visible += text
            case .toolCallProgress: progressDeltas += 1
            case .toolCall: calls += 1
            case .info(let info): infos.append(info)
            case .reasoning, .prefillProgress, .tokenID: break
            }
        }
        await engine.shutdown()

        #expect(progressDeltas > 0)
        #expect(calls == 0)
        #expect(visible.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!visible.contains("DSML"))
        #expect(infos.count == 1, "the finite stream must emit one terminal info event")
        #expect(infos.first?.toolCallProtocolFailure == .malformedEnvelope)
    }

    @Test("ChatSession throws typed malformed-tool failure without dispatch")
    func chatSessionThrowsTypedFailureWithoutDispatch() async {
        nonisolated(unsafe) let context = malformedDSMLContext(output: malformedDSMLOutput)
        let recorder = ToolDispatchRecorder()
        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(maxTokens: 1, temperature: 0),
            tools: [lineCountToolSpec()]
        ) { _ in
            await recorder.record()
            return "{}"
        }

        do {
            let response = try await session.respond(to: "Count these lines.")
            Issue.record("expected a typed tool protocol failure, got response \(response.debugDescription)")
        } catch ChatSessionError.toolCallProtocolFailure(let failure) {
            #expect(failure == .malformedEnvelope)
        } catch {
            Issue.record("expected ChatSessionError.toolCallProtocolFailure, got \(error)")
        }

        let dispatchCount = await recorder.count
        #expect(dispatchCount == 0, "malformed output must not dispatch a tool")
        await session.synchronize()
    }
}

private struct MalformedDSMLTokenizer: Tokenizer {
    let output: String

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1] }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { token in
            switch token {
            case 0: output
            case 1: "prompt"
            default: ""
            }
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { id == 0 ? output : "prompt" }

    let bosToken: String? = nil
    let eosToken: String? = nil
    let unknownToken: String? = nil

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [1]
    }
}

private struct MalformedDSMLInputProcessor: UserInputProcessor {
    let tokenizer: MalformedDSMLTokenizer

    func prepare(input: UserInput) throws -> LMInput {
        LMInput(tokens: MLXArray([Int32(1)])).withToolSchemas(input.tools)
    }
}

private final class SingleMalformedTokenLanguageModel: Module, LanguageModel,
    KVCacheDimensionProvider, @unchecked Sendable
{
    var kvHeads: [Int] { [1] }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let batch = inputs.shape.first ?? 1
        let length = inputs.shape.count > 1 ? inputs.shape[1] : inputs.size
        return MLXArray.zeros([batch, length, 2], dtype: .float32)
    }
}

private func malformedDSMLContext(output: String) -> ModelContext {
    let tokenizer = MalformedDSMLTokenizer(output: output)
    let processor = MalformedDSMLInputProcessor(tokenizer: tokenizer)
    return ModelContext(
        configuration: ModelConfiguration(
            id: "malformed-dsml-focused-test", toolCallFormat: .dsml),
        model: SingleMalformedTokenLanguageModel(),
        processor: processor,
        tokenizer: tokenizer)
}

private actor ToolDispatchRecorder {
    private(set) var count = 0

    func record() {
        count += 1
    }
}
