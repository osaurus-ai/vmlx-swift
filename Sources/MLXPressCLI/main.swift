import Foundation
import MLXLMCommon
import MLXPress

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

@main
struct MLXPressCLI {
    static func main() async {
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            if args.contains("--runtime-check") {
                let jsonOutput = args.contains("--json")
                try printRuntimeCapabilitiesJSONOrText(jsonOutput: jsonOutput)
                return
            }
            let options = try Options(args: args)
            applyFileReadPressureDefaults(options)
            applyActiveExpertTraceDefaults(options)
            let restoreCompiledDecodeEnv = applyCompiledDecodeDefaults(options)
            defer { restoreCompiledDecodeEnv() }
            let metricsWriter = try options.metricsJSONLURL.map { try MetricsJSONLWriter(url: $0) }
            let preRunPressure = metricsWriter == nil ? nil : SystemPressureSnapshot.current()

            if options.inspectOnly {
                let facts = MLXPressBundleFacts.inspect(at: options.modelDirectory)
                let readiness = MLXPressModelReadinessChecklist.build(for: facts)
                let memory = options.printMemory ? MLXPressMemorySnapshot.current() : nil
                if options.jsonOutput {
                    try printInspectJSON(facts: facts, readiness: readiness, memory: memory)
                } else {
                    printBundleFacts(facts)
                    printReadinessChecklist(readiness)
                    if let memory {
                        printMemorySnapshot(memory, label: "Memory")
                    }
                }
                return
            }

            let shouldSampleMemory = options.printMemory || options.activityGatePercent != nil
            let preLoadMemory = shouldSampleMemory
                ? MLXPressMemorySnapshot.current()
                : nil
            let preflightFacts = MLXPressBundleFacts.inspect(at: options.modelDirectory)
            guard preflightFacts.totalSafetensorsBytes > 0 else {
                throw CLIError.loadRequiresLocalWeights
            }
            if let preRunPressure {
                try metricsWriter?.writeSystemPressure(
                    phase: "pre_load",
                    snapshot: preRunPressure,
                    baseline: nil)
            }
            let peakTracker = shouldSampleMemory ? MemoryPeakTracker() : nil
            if let preLoadMemory {
                peakTracker?.record(preLoadMemory)
            }
            peakTracker?.start()

            let session = try await MLXPressSession.load(
                from: options.modelDirectory,
                configuration: options.loadConfiguration)
            let postLoadMemory = shouldSampleMemory
                ? session.memorySnapshot()
                : nil
            if let postLoadMemory {
                peakTracker?.record(postLoadMemory)
            }
            if let snapshot = SystemPressureSnapshot.current() {
                try metricsWriter?.writeSystemPressure(
                    phase: "post_load",
                    snapshot: snapshot,
                    baseline: preRunPressure)
            }

            if options.printMemory, let preLoadMemory, let postLoadMemory {
                printMemorySnapshot(preLoadMemory, label: "Pre-load memory")
                printMemorySnapshot(postLoadMemory, label: "Post-load memory")
                printMemoryDelta(
                    preLoad: preLoadMemory,
                    postLoad: postLoadMemory,
                    modelBytes: session.bundleFacts.totalSafetensorsBytes,
                    label: "Post-load memory delta")
            }

            let status = session.status()
            if status.enabled {
                fputs(
                    "MLXPress enabled backend=\(status.backend) cold=\(status.coldFraction ?? 0) tiles=\(status.tilesUnderManagement) routed-bytes=\(MLXPressFormatBytes(status.totalRoutedBytes))\n",
                    stderr)
            }
            let cacheStatus = session.cacheStatus()
            if cacheStatus.enabled {
                let maxKV = cacheStatus.defaultMaxKVSize.map(String.init) ?? "none"
                fputs(
                    "MLXPress cache-stack enabled paged=\(cacheStatus.pagedCacheEnabled) disk=\(cacheStatus.diskCacheEnabled) hybrid=\(cacheStatus.hybrid) paged-incompatible=\(cacheStatus.pagedIncompatible) default-kv=\(cacheStatus.defaultKVMode) default-max-kv=\(maxKV)\n",
                    stderr)
            }

            var resolvedGenerateParameters = await session.container.defaultGenerateParameters(
                fallback: GenerateParameters(
                    maxTokens: options.maxTokens,
                    prefillStepSize: options.prefillStepSize))
            resolvedGenerateParameters.maxTokens = options.maxTokens
            resolvedGenerateParameters.prefillStepSize = options.prefillStepSize
            if let temperature = options.temperature {
                resolvedGenerateParameters.temperature = temperature
            }
            if let topP = options.topP {
                resolvedGenerateParameters.topP = topP
            }
            resolvedGenerateParameters.enableCompiledDecode = options.enableCompiledDecode
            resolvedGenerateParameters.compiledMaxCacheLength = options.compiledMaxCacheLength

            try metricsWriter?.write([
                "type": "run_start",
                "timestamp": isoTimestamp(),
                "model_path": options.modelDirectory.path,
                "model_name": options.modelDirectory.lastPathComponent,
                "model_bytes": jsonNumber(preflightFacts.totalSafetensorsBytes),
                "turn_count": options.turns.count,
                "max_tokens": options.maxTokens,
                "prefill_step_size": resolvedGenerateParameters.prefillStepSize,
                "temperature": jsonFloat(resolvedGenerateParameters.temperature),
                "top_p": jsonFloat(resolvedGenerateParameters.topP),
                "thinking": options.enableThinking.map { NSNumber(value: $0) } ?? NSNull(),
                "compiled_decode": options.enableCompiledDecode,
                "compiled_max_cache_length": jsonOptionalNumber(
                    options.compiledMaxCacheLength.map(UInt64.init)),
                "allow_minimax_compiled_decode": options.allowMiniMaxCompiledDecode,
                "file_read_gate_mb_per_generated_token": jsonOptionalDouble(
                    options.fileReadGateMBPerGeneratedToken),
                "active_expert_trace": options.activeExpertTrace,
            ])

            if let maxFootprintPercent = options.activityGatePercent,
                let preLoadMemory,
                let postLoadMemory
            {
                let check = MLXPressActivityCompressionCheck(
                    bundleFacts: session.bundleFacts,
                    preLoad: preLoadMemory,
                    postLoad: postLoadMemory,
                    maxFootprintPercent: maxFootprintPercent)
                printActivityCompressionCheck(
                    check,
                    label: "Post-load Activity Monitor compression gate")
                if !check.passed, options.activityGateStopOnFailure {
                    await peakTracker?.stop()
                    exit(2)
                }
            }

            if let maxFootprintPercent = options.activityGatePercent,
                preLoadMemory == nil
            {
                fputs(
                    "Post-load Activity Monitor compression gate: verdict=unavailable reason=missing-preload-footprint gate=\(formatPercent(maxFootprintPercent))\n",
                    stderr)
            }

            if let maxFootprintPercent = options.activityGatePercent,
                let preLoadMemory
            {
                peakTracker?.armActivityGate(
                    bundleFacts: session.bundleFacts,
                    preLoad: preLoadMemory,
                    maxFootprintPercent: maxFootprintPercent,
                    terminateProcessOnFailure: options.activityGateStopOnFailure)
            }

            var allTurnsCoherent = true
            var allFileReadPressurePassed = true
            var totalGeneratedTokens = 0
            var lastFileReadPressure: FileReadPressureReport?
            var chatHistory: [Chat.Message] = []
            for (turnIndex, turnPrompt) in options.turns.enumerated() {
                chatHistory.append(.user(turnPrompt))
                var assistantText = ""
                var reasoningText = ""
                var completionInfo: GenerateCompletionInfo?
                nonisolated(unsafe) let userInput = UserInput(
                    chat: chatHistory,
                    additionalContext: options.chatTemplateContext)
                let parameters = resolvedGenerateParameters
                let stream = try await session.generate(
                    input: userInput,
                    parameters: parameters)

                for await event in stream {
                    switch event {
                    case .chunk(let text):
                        assistantText += text
                        print(text, terminator: "")
                        fflush(stdout)
                    case .reasoning(let text):
                        reasoningText += text
                    case .prefillProgress:
                        break
                    case .toolCall:
                        break
                    case .info(let info):
                        completionInfo = info
                    }
                }
                print("")
                if let completionInfo {
                    printGenerationTelemetry(completionInfo, turn: turnIndex + 1)
                } else {
                    fputs("Generation telemetry turn=\(turnIndex + 1) unavailable\n", stderr)
                }
                let coherencyReport = printTurnCoherency(
                    visibleText: assistantText,
                    reasoningText: reasoningText,
                    completionInfo: completionInfo,
                    expectedText: options.expectedOutputs[safe: turnIndex],
                    turn: turnIndex + 1,
                    minVisibleChars: options.minVisibleChars,
                    minGenerationTokens: options.minGenerationTokens,
                    failOnLengthStop: options.failOnLengthStop)
                let turnCoherent = coherencyReport.passed
                allTurnsCoherent = allTurnsCoherent && turnCoherent
                totalGeneratedTokens += completionInfo?.generationTokenCount ?? 0
                try metricsWriter?.writeTurn(
                    turn: turnIndex + 1,
                    modelDirectory: options.modelDirectory,
                    completionInfo: completionInfo,
                    coherency: coherencyReport,
                    visibleText: assistantText,
                    reasoningText: reasoningText)
                chatHistory.append(.assistant(assistantText))
                MLXPressStreamingProfile.dump(reason: "turn=\(turnIndex + 1)-end")
                if MLXPressStreamingProfile.isEnabled {
                    let snapshot = MLXPressStreamingProfile.snapshot()
                    try metricsWriter?.writeStreamingProfile(
                        reason: "turn=\(turnIndex + 1)-end",
                        snapshot: snapshot)
                    let residency = MLXPressActiveSliceResidency.snapshot()
                    if residency.isActive {
                        try metricsWriter?.writeActiveSliceResidency(
                            reason: "turn=\(turnIndex + 1)-end",
                            snapshot: residency)
                    }
                    let fileReadPressure = FileReadPressureReport(
                        turn: turnIndex + 1,
                        generatedTokens: totalGeneratedTokens,
                        maxMBPerGeneratedToken: options.fileReadGateMBPerGeneratedToken,
                        snapshot: snapshot)
                    printFileReadPressure(fileReadPressure)
                    try metricsWriter?.writeFileReadPressure(fileReadPressure)
                    lastFileReadPressure = fileReadPressure
                    allFileReadPressurePassed =
                        allFileReadPressurePassed && fileReadPressure.passed
                }
                if MLXPressActiveExpertTrace.isEnabled {
                    let snapshot = MLXPressActiveExpertTrace.snapshot()
                    MLXPressActiveExpertTrace.dump(reason: "turn=\(turnIndex + 1)-end")
                    try metricsWriter?.writeActiveExpertTrace(
                        reason: "turn=\(turnIndex + 1)-end",
                        snapshot: snapshot)
                }
                MLXPressClearMemoryCache()
            }

            let postDecodeMemory = shouldSampleMemory
                ? session.memorySnapshot()
                : nil
            if let postDecodeMemory {
                peakTracker?.record(postDecodeMemory)
            }
            await peakTracker?.stop()
            let peakMemory = peakTracker?.peakSnapshot()
            if options.printMemory, let postDecodeMemory {
                printMemorySnapshot(postDecodeMemory, label: "Post-decode memory")
                if let preLoadMemory {
                    printMemoryDelta(
                        preLoad: preLoadMemory,
                        postLoad: postDecodeMemory,
                        modelBytes: session.bundleFacts.totalSafetensorsBytes,
                        label: "Post-decode memory delta")
                }
            }
            if let postDecodeMemory {
                try metricsWriter?.writeMemory(
                    phase: "post_decode",
                    snapshot: postDecodeMemory,
                    preLoad: preLoadMemory,
                    modelBytes: session.bundleFacts.totalSafetensorsBytes)
            }
            if options.printMemory, let peakMemory {
                printMemorySnapshot(peakMemory, label: "Peak memory")
                if let preLoadMemory {
                    printMemoryDelta(
                        preLoad: preLoadMemory,
                        postLoad: peakMemory,
                        modelBytes: session.bundleFacts.totalSafetensorsBytes,
                        label: "Peak memory delta")
                }
            }
            if let peakMemory {
                try metricsWriter?.writeMemory(
                    phase: "peak",
                    snapshot: peakMemory,
                    preLoad: preLoadMemory,
                    modelBytes: session.bundleFacts.totalSafetensorsBytes)
            }
            let postDecodePressure = SystemPressureSnapshot.current()
            if let snapshot = postDecodePressure {
                try metricsWriter?.writeSystemPressure(
                    phase: "post_decode",
                    snapshot: snapshot,
                    baseline: preRunPressure)
            }
            if let preRunPressure,
                let postDecodePressure,
                let maxReadMB = options.fileReadGateMBPerGeneratedToken
            {
                let effectiveReadPressure = EffectiveReadPressureReport(
                    turn: options.turns.count,
                    generatedTokens: totalGeneratedTokens,
                    maxMBPerGeneratedToken: maxReadMB,
                    explicitReadBytes: lastFileReadPressure?.readBytes ?? 0,
                    preRunPressure: preRunPressure,
                    postDecodePressure: postDecodePressure)
                printEffectiveReadPressure(effectiveReadPressure)
                try metricsWriter?.writeEffectiveReadPressure(effectiveReadPressure)
                allFileReadPressurePassed =
                    allFileReadPressurePassed && effectiveReadPressure.passed
            }

            if let maxFootprintPercent = options.activityGatePercent,
                let preLoadMemory,
                let postDecodeMemory
            {
                let check = MLXPressActivityCompressionCheck(
                    bundleFacts: session.bundleFacts,
                    preLoad: preLoadMemory,
                    postLoad: postDecodeMemory,
                    maxFootprintPercent: maxFootprintPercent)
                printActivityCompressionCheck(
                    check,
                    label: "Post-decode Activity Monitor compression gate")
                try metricsWriter?.writeActivityGate(check, phase: "post_decode")
            }

            let peakActivityCheck: MLXPressActivityCompressionCheck?
            if let maxFootprintPercent = options.activityGatePercent,
                let preLoadMemory,
                let peakMemory
            {
                let check = MLXPressActivityCompressionCheck(
                    bundleFacts: session.bundleFacts,
                    preLoad: preLoadMemory,
                    postLoad: peakMemory,
                    maxFootprintPercent: maxFootprintPercent)
                printActivityCompressionCheck(
                    check,
                    label: "Peak Activity Monitor compression gate")
                try metricsWriter?.writeActivityGate(check, phase: "peak")
                peakActivityCheck = check
            } else {
                peakActivityCheck = nil
            }

            if let peakActivityCheck,
                !peakActivityCheck.passed,
                options.activityGateStopOnFailure
            {
                exit(2)
            }
            if options.requiresStrictCoherency, !allTurnsCoherent {
                exit(3)
            }
            if options.fileReadGateStopOnFailure,
                options.fileReadGateMBPerGeneratedToken != nil,
                !allFileReadPressurePassed
            {
                exit(4)
            }
        } catch {
            fputs("\(error)\n\n\(Options.usage)\n", stderr)
            exit(1)
        }
    }
}

private final class MemoryPeakTracker: @unchecked Sendable {
    private struct ActivityGate: Sendable {
        let bundleFacts: MLXPressBundleFacts
        let preLoad: MLXPressMemorySnapshot
        let maxFootprintPercent: Double
        let terminateProcessOnFailure: Bool
    }

    private let lock = NSLock()
    private var peakResidentSizeBytes: UInt64 = 0
    private var peakPhysicalFootprintBytes: UInt64?
    private var physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory
    private var peakMLXActiveMemoryBytes: UInt64 = 0
    private var peakMLXCacheMemoryBytes: UInt64 = 0
    private var peakMLXPeakMemoryBytes: UInt64 = 0
    private var activityGate: ActivityGate?
    private var reportedActivityGateFailure = false
    private var sampler: Task<Void, Never>?

    func start(intervalNanoseconds: UInt64 = 250_000_000) {
        sampler = Task.detached { [weak self] in
            while !Task.isCancelled {
                self?.record(MLXPressMemorySnapshot.current())
                try? await Task.sleep(nanoseconds: intervalNanoseconds)
            }
        }
    }

    func stop() async {
        if let sampler {
            sampler.cancel()
            await sampler.value
        }
        record(MLXPressMemorySnapshot.current())
    }

    func record(_ snapshot: MLXPressMemorySnapshot) {
        let failure: MLXPressActivityCompressionCheck?
        let shouldTerminate: Bool
        lock.lock()
        peakResidentSizeBytes = max(
            peakResidentSizeBytes,
            snapshot.residentSizeBytes)
        if let footprint = snapshot.physicalFootprintBytes {
            peakPhysicalFootprintBytes = max(
                peakPhysicalFootprintBytes ?? 0,
                footprint)
        }
        physicalMemoryBytes = snapshot.physicalMemoryBytes
        peakMLXActiveMemoryBytes = max(
            peakMLXActiveMemoryBytes,
            snapshot.mlxActiveMemoryBytes)
        peakMLXCacheMemoryBytes = max(
            peakMLXCacheMemoryBytes,
            snapshot.mlxCacheMemoryBytes)
        peakMLXPeakMemoryBytes = max(
            peakMLXPeakMemoryBytes,
            snapshot.mlxPeakMemoryBytes)
        if let activityGate,
            !reportedActivityGateFailure
        {
            let check = MLXPressActivityCompressionCheck(
                bundleFacts: activityGate.bundleFacts,
                preLoad: activityGate.preLoad,
                postLoad: snapshot,
                maxFootprintPercent: activityGate.maxFootprintPercent)
            if check.verdict == .failed {
                reportedActivityGateFailure = true
                failure = check
                shouldTerminate = activityGate.terminateProcessOnFailure
            } else {
                failure = nil
                shouldTerminate = false
            }
        } else {
            failure = nil
            shouldTerminate = false
        }
        lock.unlock()

        if let failure {
            printActivityCompressionCheck(
                failure,
                label: "Peak Activity Monitor compression gate")
            fflush(stderr)
            if shouldTerminate {
                terminateImmediately(2)
            }
        }
    }

    func armActivityGate(
        bundleFacts: MLXPressBundleFacts,
        preLoad: MLXPressMemorySnapshot,
        maxFootprintPercent: Double,
        terminateProcessOnFailure: Bool
    ) {
        lock.lock()
        activityGate = ActivityGate(
            bundleFacts: bundleFacts,
            preLoad: preLoad,
            maxFootprintPercent: maxFootprintPercent,
            terminateProcessOnFailure: terminateProcessOnFailure)
        reportedActivityGateFailure = false
        let peak = MLXPressMemorySnapshot(
            residentSizeBytes: peakResidentSizeBytes,
            physicalFootprintBytes: peakPhysicalFootprintBytes,
            physicalMemoryBytes: physicalMemoryBytes,
            mlxActiveMemoryBytes: peakMLXActiveMemoryBytes,
            mlxCacheMemoryBytes: peakMLXCacheMemoryBytes,
            mlxPeakMemoryBytes: peakMLXPeakMemoryBytes)
        lock.unlock()
        record(peak)
    }

    func peakSnapshot() -> MLXPressMemorySnapshot {
        lock.lock()
        let snapshot = MLXPressMemorySnapshot(
            residentSizeBytes: peakResidentSizeBytes,
            physicalFootprintBytes: peakPhysicalFootprintBytes,
            physicalMemoryBytes: physicalMemoryBytes,
            mlxActiveMemoryBytes: peakMLXActiveMemoryBytes,
            mlxCacheMemoryBytes: peakMLXCacheMemoryBytes,
            mlxPeakMemoryBytes: peakMLXPeakMemoryBytes)
        lock.unlock()
        return snapshot
    }
}

private func terminateImmediately(_ status: Int32) -> Never {
    #if canImport(Darwin) || canImport(Glibc)
    _exit(status)
    #else
    exit(Int(status))
    #endif
}

private let fileReadPressureRowNames: Set<String> = [
    "tensor.read",
    "tensor.stacked_read",
    "tensor.stacked_bank_read",
    "tensor.mach_offset_read",
]

private struct FileReadPressureReport {
    let turn: Int
    let generatedTokens: Int
    let readBytes: UInt64
    let readRows: [String]
    let maxMBPerGeneratedToken: Double?

    init(
        turn: Int,
        generatedTokens: Int,
        maxMBPerGeneratedToken: Double?,
        snapshot: MLXPressStreamingProfileSnapshot
    ) {
        self.turn = turn
        self.generatedTokens = generatedTokens
        self.maxMBPerGeneratedToken = maxMBPerGeneratedToken
        var bytes: UInt64 = 0
        var rows: [String] = []
        for row in snapshot.rows where fileReadPressureRowNames.contains(row.name) {
            bytes += UInt64(max(0, row.bytes))
            rows.append(row.name)
        }
        self.readBytes = bytes
        self.readRows = Array(Set(rows)).sorted()
    }

    var readMB: Double {
        Double(readBytes) / (1024 * 1024)
    }

    var readMBPerGeneratedToken: Double {
        readMB / Double(max(1, generatedTokens))
    }

    var passed: Bool {
        guard let maxMBPerGeneratedToken else { return true }
        guard generatedTokens > 0 else { return readBytes == 0 }
        return readMBPerGeneratedToken <= maxMBPerGeneratedToken
    }

    var verdict: String {
        guard maxMBPerGeneratedToken != nil else { return "unconfigured" }
        return passed ? "passed" : "failed"
    }
}

private func printFileReadPressure(_ report: FileReadPressureReport) {
    var parts = [
        "turn=\(report.turn)",
        "scope=run-so-far",
        "read=\(MLXPressFormatBytes(report.readBytes))",
        "read-mb=\(formatDecimal(report.readMB))",
        "read-mb/generated-token=\(formatDecimal(report.readMBPerGeneratedToken))",
        "generated-tokens=\(report.generatedTokens)",
        "rows=\(report.readRows.isEmpty ? "none" : report.readRows.joined(separator: ","))",
        "verdict=\(report.verdict)",
    ]
    if let gate = report.maxMBPerGeneratedToken {
        parts.append("gate-mb/generated-token=\(formatDecimal(gate))")
    }
    fputs("File-read pressure \(parts.joined(separator: " "))\n", stderr)
}

private struct EffectiveReadPressureReport {
    let turn: Int
    let generatedTokens: Int
    let maxMBPerGeneratedToken: Double
    let explicitReadBytes: UInt64
    let pageinBytes: UInt64

    init(
        turn: Int,
        generatedTokens: Int,
        maxMBPerGeneratedToken: Double,
        explicitReadBytes: UInt64,
        preRunPressure: SystemPressureSnapshot,
        postDecodePressure: SystemPressureSnapshot
    ) {
        self.turn = turn
        self.generatedTokens = generatedTokens
        self.maxMBPerGeneratedToken = maxMBPerGeneratedToken
        self.explicitReadBytes = explicitReadBytes
        let pageins = counterDelta(postDecodePressure.pageins, preRunPressure.pageins)
        self.pageinBytes = pageins * postDecodePressure.pageSizeBytes
    }

    var effectiveReadBytes: UInt64 {
        max(explicitReadBytes, pageinBytes)
    }

    var effectiveReadMB: Double {
        Double(effectiveReadBytes) / (1024 * 1024)
    }

    var effectiveReadMBPerGeneratedToken: Double {
        effectiveReadMB / Double(max(1, generatedTokens))
    }

    var explicitReadMB: Double {
        Double(explicitReadBytes) / (1024 * 1024)
    }

    var pageinMB: Double {
        Double(pageinBytes) / (1024 * 1024)
    }

    var passed: Bool {
        guard generatedTokens > 0 else { return effectiveReadBytes == 0 }
        return effectiveReadMBPerGeneratedToken <= maxMBPerGeneratedToken
    }

    var verdict: String {
        passed ? "passed" : "failed"
    }
}

private func printEffectiveReadPressure(_ report: EffectiveReadPressureReport) {
    let parts = [
        "turn=\(report.turn)",
        "scope=run-so-far",
        "effective-read=\(MLXPressFormatBytes(report.effectiveReadBytes))",
        "effective-read-mb/generated-token=\(formatDecimal(report.effectiveReadMBPerGeneratedToken))",
        "explicit-read-mb=\(formatDecimal(report.explicitReadMB))",
        "pagein-mb=\(formatDecimal(report.pageinMB))",
        "generated-tokens=\(report.generatedTokens)",
        "verdict=\(report.verdict)",
        "gate-mb/generated-token=\(formatDecimal(report.maxMBPerGeneratedToken))",
    ]
    fputs("Effective read pressure \(parts.joined(separator: " "))\n", stderr)
}

private func applyFileReadPressureDefaults(_ options: Options) {
    guard options.fileReadGateMBPerGeneratedToken != nil else { return }
    if getenv("MLXPRESS_STREAMING_PROFILE") == nil {
        setenv("MLXPRESS_STREAMING_PROFILE", "1", 1)
    }
    if getenv("MLXPRESS_STREAMING_PROFILE_EVERY") == nil {
        setenv("MLXPRESS_STREAMING_PROFILE_EVERY", "0", 1)
    }
}

private func applyActiveExpertTraceDefaults(_ options: Options) {
    guard options.activeExpertTrace else { return }
    setenv("MLXPRESS_ACTIVE_EXPERT_TRACE", "1", 1)
    if getenv("MLXPRESS_STREAMING_PROFILE") == nil {
        setenv("MLXPRESS_STREAMING_PROFILE", "1", 1)
    }
    if getenv("MLXPRESS_STREAMING_PROFILE_EVERY") == nil {
        setenv("MLXPRESS_STREAMING_PROFILE_EVERY", "0", 1)
    }
}

private func applyCompiledDecodeDefaults(_ options: Options) -> () -> Void {
    let keys = [
        "MLXPRESS_COMPILED_DECODE_ALLOW_MINIMAX",
        "JANGPRESS_COMPILED_DECODE_ALLOW_MINIMAX",
    ]
    let previous = Dictionary(uniqueKeysWithValues: keys.map {
        ($0, getenvString($0))
    })
    if options.allowMiniMaxCompiledDecode {
        for key in keys {
            setenv(key, "1", 1)
        }
    }
    return {
        for key in keys {
            if let value = previous[key] ?? nil {
                setenv(key, value, 1)
            } else {
                unsetenv(key)
            }
        }
    }
}

private func getenvString(_ key: String) -> String? {
    guard let raw = getenv(key) else { return nil }
    return String(cString: raw)
}

private struct SystemPressureSnapshot {
    let pageSizeBytes: UInt64
    let pageins: UInt64
    let pageouts: UInt64
    let swapins: UInt64
    let swapouts: UInt64
    let compressions: UInt64
    let decompressions: UInt64

    static func current() -> SystemPressureSnapshot? {
        #if canImport(Darwin)
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        var stats = vm_statistics64_data_t()
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(
                    mach_host_self(),
                    HOST_VM_INFO64,
                    $0,
                    &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        var pageSize: vm_size_t = 0
        guard host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS else {
            return nil
        }
        return SystemPressureSnapshot(
            pageSizeBytes: UInt64(pageSize),
            pageins: UInt64(stats.pageins),
            pageouts: UInt64(stats.pageouts),
            swapins: UInt64(stats.swapins),
            swapouts: UInt64(stats.swapouts),
            compressions: UInt64(stats.compressions),
            decompressions: UInt64(stats.decompressions))
        #else
        return nil
        #endif
    }

    func jsonObject(baseline: SystemPressureSnapshot?) -> [String: Any] {
        var object: [String: Any] = [
            "page_size_bytes": jsonNumber(pageSizeBytes),
            "pageins": jsonNumber(pageins),
            "pageouts": jsonNumber(pageouts),
            "swapins": jsonNumber(swapins),
            "swapouts": jsonNumber(swapouts),
            "compressions": jsonNumber(compressions),
            "decompressions": jsonNumber(decompressions),
        ]
        if let baseline {
            let pageinDelta = counterDelta(pageins, baseline.pageins)
            let pageoutDelta = counterDelta(pageouts, baseline.pageouts)
            let swapinDelta = counterDelta(swapins, baseline.swapins)
            let swapoutDelta = counterDelta(swapouts, baseline.swapouts)
            object["pageins_delta"] = jsonNumber(pageinDelta)
            object["pageouts_delta"] = jsonNumber(pageoutDelta)
            object["swapins_delta"] = jsonNumber(swapinDelta)
            object["swapouts_delta"] = jsonNumber(swapoutDelta)
            object["pageins_delta_bytes"] = jsonNumber(pageinDelta * pageSizeBytes)
            object["pageouts_delta_bytes"] = jsonNumber(pageoutDelta * pageSizeBytes)
            object["swapins_delta_bytes"] = jsonNumber(swapinDelta * pageSizeBytes)
            object["swapouts_delta_bytes"] = jsonNumber(swapoutDelta * pageSizeBytes)
            object["compressions_delta"] = jsonNumber(
                counterDelta(compressions, baseline.compressions))
            object["decompressions_delta"] = jsonNumber(
                counterDelta(decompressions, baseline.decompressions))
        }
        return object
    }
}

private final class MetricsJSONLWriter {
    private let handle: FileHandle
    private let lock = NSLock()

    init(url: URL) throws {
        let parent = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: parent,
            withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: 0)
    }

    deinit {
        try? handle.close()
    }

    func write(_ object: [String: Any]) throws {
        var payload = object
        if payload["timestamp"] == nil {
            payload["timestamp"] = isoTimestamp()
        }
        guard JSONSerialization.isValidJSONObject(payload) else {
            throw CLIError.invalidMetricsObject
        }
        var data = try JSONSerialization.data(
            withJSONObject: payload,
            options: [.sortedKeys])
        data.append(0x0a)
        lock.lock()
        defer { lock.unlock() }
        try handle.write(contentsOf: data)
    }

    func writeTurn(
        turn: Int,
        modelDirectory: URL,
        completionInfo: GenerateCompletionInfo?,
        coherency: TurnCoherencyReport,
        visibleText: String,
        reasoningText: String
    ) throws {
        var object: [String: Any] = [
            "type": "turn",
            "turn": turn,
            "model_name": modelDirectory.lastPathComponent,
            "visible_chars": coherency.visibleChars,
            "reasoning_chars": coherency.reasoningChars,
            "visible_preview": String(visibleText.trimmingCharacters(in: .whitespacesAndNewlines).prefix(160)),
            "reasoning_preview": String(reasoningText.trimmingCharacters(in: .whitespacesAndNewlines).prefix(160)),
            "coherency": coherency.jsonObject,
        ]
        if let completionInfo {
            object["telemetry"] = [
                "prompt_tokens": completionInfo.promptTokenCount,
                "prompt_tokens_per_second": jsonDouble(completionInfo.promptTokensPerSecond),
                "generation_tokens": completionInfo.generationTokenCount,
                "tokens_per_second": jsonDouble(completionInfo.tokensPerSecond),
                "prompt_time_seconds": jsonDouble(completionInfo.promptTime),
                "generation_time_seconds": jsonDouble(completionInfo.generateTime),
                "stop_reason": String(describing: completionInfo.stopReason),
                "unclosed_reasoning": completionInfo.unclosedReasoning,
            ]
        } else {
            object["telemetry"] = NSNull()
        }
        try write(object)
    }

    func writeMemory(
        phase: String,
        snapshot: MLXPressMemorySnapshot,
        preLoad: MLXPressMemorySnapshot?,
        modelBytes: UInt64
    ) throws {
        var object: [String: Any] = [
            "type": "memory",
            "phase": phase,
            "resident_size_bytes": jsonNumber(snapshot.residentSizeBytes),
            "physical_memory_bytes": jsonNumber(snapshot.physicalMemoryBytes),
            "mlx_active_memory_bytes": jsonNumber(snapshot.mlxActiveMemoryBytes),
            "mlx_cache_memory_bytes": jsonNumber(snapshot.mlxCacheMemoryBytes),
            "mlx_peak_memory_bytes": jsonNumber(snapshot.mlxPeakMemoryBytes),
            "model_bytes": jsonNumber(modelBytes),
        ]
        object["physical_footprint_bytes"] = jsonOptionalNumber(snapshot.physicalFootprintBytes)
        if let preLoad {
            object["resident_delta_bytes"] = signedDelta(
                snapshot.residentSizeBytes,
                preLoad.residentSizeBytes)
            object["mlx_active_delta_bytes"] = signedDelta(
                snapshot.mlxActiveMemoryBytes,
                preLoad.mlxActiveMemoryBytes)
            object["mlx_cache_delta_bytes"] = signedDelta(
                snapshot.mlxCacheMemoryBytes,
                preLoad.mlxCacheMemoryBytes)
            object["mlx_peak_delta_bytes"] = signedDelta(
                snapshot.mlxPeakMemoryBytes,
                preLoad.mlxPeakMemoryBytes)
            if let before = preLoad.physicalFootprintBytes,
                let after = snapshot.physicalFootprintBytes
            {
                let delta = signedDelta(after, before)
                object["physical_footprint_delta_bytes"] = delta
                if modelBytes > 0 {
                    object["physical_footprint_delta_percent_of_model"] =
                        Double(max(delta, 0)) / Double(modelBytes) * 100.0
                }
            }
        }
        try write(object)
    }

    func writeSystemPressure(
        phase: String,
        snapshot: SystemPressureSnapshot,
        baseline: SystemPressureSnapshot?
    ) throws {
        var object = snapshot.jsonObject(baseline: baseline)
        object["type"] = "system_pressure"
        object["phase"] = phase
        try write(object)
    }

    func writeFileReadPressure(_ report: FileReadPressureReport) throws {
        try write([
            "type": "file_read_pressure",
            "turn": report.turn,
            "scope": "run_so_far",
            "read_bytes": jsonNumber(report.readBytes),
            "read_mb": jsonDouble(report.readMB),
            "read_mb_per_generated_token": jsonDouble(report.readMBPerGeneratedToken),
            "generated_tokens": report.generatedTokens,
            "max_read_mb_per_generated_token": jsonOptionalDouble(
                report.maxMBPerGeneratedToken),
            "passed": report.passed,
            "verdict": report.verdict,
            "rows": report.readRows,
        ])
    }

    func writeEffectiveReadPressure(_ report: EffectiveReadPressureReport) throws {
        try write([
            "type": "effective_read_pressure",
            "turn": report.turn,
            "scope": "run_so_far",
            "effective_read_bytes": jsonNumber(report.effectiveReadBytes),
            "effective_read_mb": jsonDouble(report.effectiveReadMB),
            "effective_read_mb_per_generated_token": jsonDouble(
                report.effectiveReadMBPerGeneratedToken),
            "explicit_read_bytes": jsonNumber(report.explicitReadBytes),
            "explicit_read_mb": jsonDouble(report.explicitReadMB),
            "pagein_bytes": jsonNumber(report.pageinBytes),
            "pagein_mb": jsonDouble(report.pageinMB),
            "generated_tokens": report.generatedTokens,
            "max_read_mb_per_generated_token": jsonDouble(report.maxMBPerGeneratedToken),
            "passed": report.passed,
            "verdict": report.verdict,
        ])
    }

    func writeActiveExpertTrace(
        reason: String,
        snapshot: MLXPressActiveExpertTraceSnapshot
    ) throws {
        guard !snapshot.isEmpty else { return }
        let layers: [[String: Any]] = snapshot.layers.map { layer in
            let experts: [[String: Any]] = layer.topExperts.map { expert in
                [
                    "expert": expert.expertIdx,
                    "count": expert.count,
                ]
            }
            return [
                "layer": layer.layerIdx,
                "calls": layer.calls,
                "tokens": layer.tokenCount,
                "routed_slots": layer.routedSlots,
                "unique_expert_touches": layer.uniqueExpertTouches,
                "repeated_within_chunk": layer.repeatedWithinChunk,
                "consecutive_reuse_touches": layer.consecutiveReuseTouches,
                "average_unique_experts_per_call": jsonDouble(
                    layer.averageUniqueExpertsPerCall),
                "consecutive_reuse_rate": jsonDouble(layer.consecutiveReuseRate),
                "top_experts": experts,
            ]
        }
        try write([
            "type": "active_expert_trace",
            "reason": reason,
            "total_calls": snapshot.totalCalls,
            "total_tokens": snapshot.totalTokens,
            "total_routed_slots": snapshot.totalRoutedSlots,
            "total_unique_expert_touches": snapshot.totalUniqueExpertTouches,
            "total_repeated_within_chunk": snapshot.totalRepeatedWithinChunk,
            "total_consecutive_reuse_touches": snapshot.totalConsecutiveReuseTouches,
            "consecutive_reuse_rate": jsonDouble(snapshot.consecutiveReuseRate),
            "layers": layers,
        ])
    }

    func writeActiveSliceResidency(
        reason: String,
        snapshot: MLXPressActiveSliceResidencySnapshot
    ) throws {
        try write([
            "type": "active_slice_residency",
            "reason": reason,
            "budget_bytes": jsonNumber(snapshot.budgetBytes),
            "tensor_budget_bytes": jsonNumber(snapshot.tensorBudgetBytes),
            "bank_budget_bytes": jsonNumber(snapshot.bankBudgetBytes),
            "resident_bytes": jsonNumber(snapshot.residentBytes),
            "tensor_resident_bytes": jsonNumber(snapshot.tensorResidentBytes),
            "slice_resident_bytes": jsonNumber(snapshot.sliceResidentBytes),
            "bank_resident_bytes": jsonNumber(snapshot.bankResidentBytes),
            "entries": snapshot.entries,
            "tensor_entries": snapshot.tensorEntries,
            "slice_entries": snapshot.sliceEntries,
            "bank_entries": snapshot.bankEntries,
            "hits": snapshot.hits,
            "tensor_hits": snapshot.tensorHits,
            "slice_hits": snapshot.sliceHits,
            "bank_hits": snapshot.bankHits,
            "misses": snapshot.misses,
            "tensor_misses": snapshot.tensorMisses,
            "slice_misses": snapshot.sliceMisses,
            "bank_misses": snapshot.bankMisses,
            "requests": snapshot.requests,
            "hit_rate": jsonDouble(snapshot.hitRate),
            "hit_bytes": jsonNumber(snapshot.hitBytes),
            "tensor_hit_bytes": jsonNumber(snapshot.tensorHitBytes),
            "slice_hit_bytes": jsonNumber(snapshot.sliceHitBytes),
            "bank_hit_bytes": jsonNumber(snapshot.bankHitBytes),
            "miss_bytes": jsonNumber(snapshot.missBytes),
            "tensor_miss_bytes": jsonNumber(snapshot.tensorMissBytes),
            "slice_miss_bytes": jsonNumber(snapshot.sliceMissBytes),
            "bank_miss_bytes": jsonNumber(snapshot.bankMissBytes),
            "byte_hit_rate": jsonDouble(snapshot.byteHitRate),
            "stores": snapshot.stores,
            "tensor_stores": snapshot.tensorStores,
            "slice_stores": snapshot.sliceStores,
            "bank_stores": snapshot.bankStores,
            "evictions": snapshot.evictions,
            "tensor_evictions": snapshot.tensorEvictions,
            "slice_evictions": snapshot.sliceEvictions,
            "bank_evictions": snapshot.bankEvictions,
        ])
    }

    func writeActivityGate(
        _ check: MLXPressActivityCompressionCheck,
        phase: String
    ) throws {
        try write([
            "type": "activity_gate",
            "phase": phase,
            "verdict": check.verdict.rawValue,
            "passed": check.passed,
            "model_bytes": jsonNumber(check.modelBytes),
            "pre_load_footprint_bytes": jsonOptionalNumber(check.preLoadFootprintBytes),
            "post_load_footprint_bytes": jsonOptionalNumber(check.postLoadFootprintBytes),
            "footprint_increase_bytes": jsonOptionalNumber(check.footprintIncreaseBytes),
            "footprint_increase_percent": jsonOptionalDouble(check.footprintIncreasePercent),
            "max_allowed_footprint_increase_bytes": jsonNumber(check.maxAllowedFootprintIncreaseBytes),
            "max_footprint_percent": check.maxFootprintPercent,
        ])
    }

    func writeStreamingProfile(
        reason: String,
        snapshot: MLXPressStreamingProfileSnapshot
    ) throws {
        guard !snapshot.isEmpty else { return }
        let rows: [[String: Any]] = snapshot.rows.map { row in
            [
                "name": row.name,
                "count": row.count,
                "seconds": jsonDouble(row.seconds),
                "milliseconds": jsonDouble(row.milliseconds),
                "average_milliseconds": jsonDouble(row.averageMilliseconds),
                "bytes": row.bytes,
                "bytes_mb": jsonDouble(row.bytesMB),
                "bytes_per_call_mb": jsonDouble(row.bytesPerCallMB),
                "bandwidth_mb_per_second": jsonDouble(row.bandwidthMBps),
            ]
        }
        try write([
            "type": "streaming_profile",
            "reason": reason,
            "total_seconds": jsonDouble(snapshot.totalSeconds),
            "total_milliseconds": jsonDouble(snapshot.totalSeconds * 1000),
            "rows": rows,
        ])
    }
}

private struct Options {
    var modelDirectory: URL
    var turns: [String]
    var maxTokens: Int
    var prefillStepSize: Int
    var temperature: Float?
    var topP: Float?
    var expectedOutputs: [String]
    var inspectOnly: Bool
    var printMemory: Bool
    var activityGatePercent: Double?
    var activityGateStopOnFailure: Bool
    var jsonOutput: Bool
    var metricsJSONLURL: URL?
    var loadConfiguration: MLXPressLoadConfiguration
    var enableCompiledDecode: Bool
    var compiledMaxCacheLength: Int?
    var allowMiniMaxCompiledDecode: Bool
    var enableThinking: Bool?
    var reasoningEffort: String?
    var minVisibleChars: Int
    var minGenerationTokens: Int
    var failOnLengthStop: Bool
    var fileReadGateMBPerGeneratedToken: Double?
    var fileReadGateStopOnFailure: Bool
    var activeExpertTrace: Bool

    var requiresStrictCoherency: Bool {
        !expectedOutputs.isEmpty
            || minVisibleChars > 0
            || minGenerationTokens > 0
            || failOnLengthStop
    }

    var chatTemplateContext: [String: any Sendable] {
        var context: [String: any Sendable] = [:]
        if let enableThinking {
            context["enable_thinking"] = enableThinking
            context["thinking"] = enableThinking
        }
        if let reasoningEffort, !reasoningEffort.isEmpty {
            context["reasoning_effort"] = reasoningEffort
        }
        return context
    }

    static let usage = """
    Usage:
      mlxpress <model-dir> <prompt> [--max-tokens N] [--prefill-step-size N] [--temp T] [--top-p P] [--thinking on|off] [--reasoning-effort VALUE] [--mlxpress off|auto|N] [--resident-load on|off] [--ephemeral-prestack on|off] [--compiled-decode on|off] [--compiled-max-cache-length N] [--allow-minimax-compiled-decode] [--active-expert-streaming on|off] [--active-expert-trace] [--cache-stack on|off] [--disk-cache on|off] [--disk-cache-dir PATH] [--kv-cache none|turboquant] [--router-advice] [--print-memory] [--activity-gate PCT] [--activity-gate-report-only] [--file-read-gate-mb-per-token N] [--file-read-gate-report-only] [--metrics-jsonl PATH] [--min-visible-chars N] [--min-generation-tokens N] [--fail-on-length-stop]
      mlxpress <model-dir> --turn TEXT --turn TEXT [--expect TEXT --expect TEXT] [--max-tokens N] [--prefill-step-size N] [--thinking on|off] [--reasoning-effort VALUE] [--mlxpress off|auto|N] [--resident-load on|off] [--ephemeral-prestack on|off] [--compiled-decode on|off] [--compiled-max-cache-length N] [--allow-minimax-compiled-decode] [--active-expert-streaming on|off] [--active-expert-trace] [--cache-stack on|off] [--disk-cache on|off] [--disk-cache-dir PATH] [--kv-cache none|turboquant] [--print-memory] [--activity-gate PCT] [--activity-gate-report-only] [--file-read-gate-mb-per-token N] [--file-read-gate-report-only] [--metrics-jsonl PATH] [--min-visible-chars N] [--min-generation-tokens N] [--fail-on-length-stop]
      mlxpress <model-dir> --inspect [--print-memory] [--json]
      mlxpress --runtime-check [--json]

    Notes:
      --activity-gate PCT fails if peak phys_footprint growth across load/decode exceeds PCT% of safetensors bytes; it does not change MLX.Memory.memoryLimit.
      --prefill-step-size N bounds prompt prefill activation peaks; default is \(MLXPressDefaultPrefillStepSize).
      --turn may be repeated to run a multi-turn chat in one loaded session.
      --expect may be repeated to require visible generated text to contain each expected answer.
      --resident-load on disables safetensors mmap for this run and loads model weights into RAM.
      --thinking is tri-state: omitted leaves the model's template default untouched; on/off explicitly sets enable_thinking.
      --reasoning-effort passes the model-family chat-template effort knob (for example no_think, low, high, max).
      For strict thinking-on validation, thinking-on validation prompts must ask for a visible final answer.
      Do not use prompts that ask the model to think privately; pair thinking-on proof rows with
      --min-visible-chars and --fail-on-length-stop so reasoning-only length stops remain failures.
      --cache-stack defaults to on: disk L2, TurboQuant KV defaulting, and long-prompt KV cap are enabled; paged RAM cache is opt-in.
      --disk-cache-dir isolates disk L2 state for cold/warm validation runs.
      --ephemeral-prestack builds a temporary routed JANGTQ overlay under the system temp directory, uses the mmap loader, and removes the overlay after mmap load with a process-exit fallback. It is a MiniMax-class resident-compute diagnostic, not a permanent prestack cache.
      --compiled-decode is diagnostic opt-in for the single-sequence compiled decode path. MiniMax remains blocked unless --allow-minimax-compiled-decode is also set, because older compiled traces diverged.
      --active-expert-streaming defaults to off. MLXPress is compression-first: mmap-backed canonical weights plus OS cold-page advice. Enable active-expert streaming only for fallback diagnostics when a bundle cannot yet use the resident/compression path safely.
      --active-expert-trace records per-layer routed expert reuse into metrics JSONL.
      --activity-gate stops generation as soon as peak memory crosses the gate unless --activity-gate-report-only is set.
      --file-read-gate-mb-per-token fails if cumulative streaming tensor file reads exceed N MB per generated token.
      --file-read-gate-report-only records file-read pressure without failing the process.
      --metrics-jsonl writes per-turn token/s, coherency, memory-gate, system-pressure, and file-read-pressure records as JSON Lines.
      --min-visible-chars and --min-generation-tokens make longer no-loop smokes fail if a turn exits too early.
      --fail-on-length-stop makes a max-token stop a coherency failure for no-loop validation.

    Examples:
      mlxpress ~/.mlxstudio/models/mlx-community/Qwen3-4B-4bit "Write one sentence about MLX."
      mlxpress /models/Kimi-K2.6-Med-JANGTQ "Hello" --mlxpress 70
      mlxpress /models/Kimi-K2.6-Med-JANGTQ "Hello" --max-tokens 1 --activity-gate 30 --print-memory
      mlxpress /models/Kimi-K2.6-Med-JANGTQ --inspect --print-memory
      mlxpress /models/Kimi-K2.6-Med-JANGTQ --inspect --json
      mlxpress --runtime-check
    """

    init(args: [String]) throws {
        guard !args.isEmpty else { throw CLIError.missingArguments }

        var positional: [String] = []
        var maxTokens = 256
        var prefillStepSize = MLXPressDefaultPrefillStepSize
        var temperature: Float?
        var topP: Float?
        var compression: MLXPressCompressionPolicy = .auto(envFallback: true)
        var useMmapSafetensors = true
        var prestack: MLXPressPrestackPolicy = .disabled
        var enableCompiledDecode = false
        var compiledMaxCacheLength: Int?
        var allowMiniMaxCompiledDecode = false
        var activeExpertStreamingOverride: Bool?
        var cacheConfiguration = MLXPressCacheConfiguration()
        var routerAdvice = false
        var inspectOnly = false
        var printMemory = false
        var activityGatePercent: Double?
        var activityGateStopOnFailure = true
        var jsonOutput = false
        var metricsJSONLURL: URL?
        var turns: [String] = []
        var expectedOutputs: [String] = []
        var enableThinking: Bool?
        var reasoningEffort: String?
        var minVisibleChars = 0
        var minGenerationTokens = 0
        var failOnLengthStop = false
        var fileReadGateMBPerGeneratedToken: Double?
        var fileReadGateStopOnFailure = true
        var activeExpertTrace = false

        var index = 0
        while index < args.count {
            let arg = args[index]
            switch arg {
            case "--max-tokens":
                index += 1
                guard index < args.count, let value = Int(args[index]) else {
                    throw CLIError.badValue("--max-tokens")
                }
                maxTokens = value
            case "--prefill-step-size", "--prefill-step":
                index += 1
                guard index < args.count,
                    let value = Int(args[index]),
                    value > 0
                else {
                    throw CLIError.badValue(arg)
                }
                prefillStepSize = value
            case "--temp", "--temperature":
                index += 1
                guard index < args.count, let value = Float(args[index]) else {
                    throw CLIError.badValue(arg)
                }
                temperature = value
            case "--top-p":
                index += 1
                guard index < args.count, let value = Float(args[index]) else {
                    throw CLIError.badValue("--top-p")
                }
                topP = value
            case "--turn":
                index += 1
                guard index < args.count, !args[index].isEmpty else {
                    throw CLIError.badValue("--turn")
                }
                turns.append(args[index])
            case "--expect":
                index += 1
                guard index < args.count, !args[index].isEmpty else {
                    throw CLIError.badValue("--expect")
                }
                expectedOutputs.append(args[index])
            case "--min-visible-chars":
                index += 1
                guard index < args.count, let value = Int(args[index]), value >= 0 else {
                    throw CLIError.badValue("--min-visible-chars")
                }
                minVisibleChars = value
            case "--min-generation-tokens":
                index += 1
                guard index < args.count, let value = Int(args[index]), value >= 0 else {
                    throw CLIError.badValue("--min-generation-tokens")
                }
                minGenerationTokens = value
            case "--fail-on-length-stop":
                failOnLengthStop = true
            case "--thinking":
                index += 1
                guard index < args.count else { throw CLIError.badValue("--thinking") }
                enableThinking = try Self.parseBool(args[index], option: "--thinking")
            case "--reasoning-effort":
                index += 1
                guard index < args.count, !args[index].isEmpty else {
                    throw CLIError.badValue("--reasoning-effort")
                }
                reasoningEffort = args[index]
            case "--enable-thinking":
                enableThinking = true
            case "--disable-thinking":
                enableThinking = false
            case "--mlxpress":
                index += 1
                guard index < args.count else { throw CLIError.badValue("--mlxpress") }
                compression = try Self.parseCompression(args[index])
            case "--resident-load":
                index += 1
                guard index < args.count else { throw CLIError.badValue("--resident-load") }
                useMmapSafetensors = !(try Self.parseBool(args[index], option: "--resident-load"))
            case "--enable-resident-load":
                useMmapSafetensors = false
            case "--disable-resident-load":
                useMmapSafetensors = true
            case "--ephemeral-prestack":
                index += 1
                guard index < args.count else { throw CLIError.badValue("--ephemeral-prestack") }
                prestack = try Self.parseBool(args[index], option: "--ephemeral-prestack")
                    ? .ephemeralTemporaryOverlay
                    : .disabled
            case "--enable-ephemeral-prestack":
                prestack = .ephemeralTemporaryOverlay
            case "--disable-ephemeral-prestack":
                prestack = .disabled
            case "--compiled-decode":
                index += 1
                guard index < args.count else {
                    throw CLIError.badValue("--compiled-decode")
                }
                enableCompiledDecode = try Self.parseBool(args[index], option: "--compiled-decode")
            case "--enable-compiled-decode":
                enableCompiledDecode = true
            case "--disable-compiled-decode":
                enableCompiledDecode = false
            case "--compiled-max-cache-length":
                index += 1
                guard index < args.count,
                      let value = Int(args[index]),
                      value > 0
                else {
                    throw CLIError.badValue("--compiled-max-cache-length")
                }
                compiledMaxCacheLength = value
            case "--allow-minimax-compiled-decode":
                allowMiniMaxCompiledDecode = true
            case "--active-expert-streaming":
                index += 1
                guard index < args.count else {
                    throw CLIError.badValue("--active-expert-streaming")
                }
                activeExpertStreamingOverride = try Self.parseBool(
                    args[index], option: "--active-expert-streaming")
            case "--enable-active-expert-streaming":
                activeExpertStreamingOverride = true
            case "--disable-active-expert-streaming":
                activeExpertStreamingOverride = false
            case "--active-expert-trace":
                activeExpertTrace = true
            case "--no-active-expert-trace":
                activeExpertTrace = false
            case "--cache-stack":
                index += 1
                guard index < args.count else { throw CLIError.badValue("--cache-stack") }
                cacheConfiguration.enabled = try Self.parseBool(args[index], option: "--cache-stack")
            case "--enable-cache-stack":
                cacheConfiguration.enabled = true
            case "--disable-cache-stack":
                cacheConfiguration.enabled = false
            case "--disk-cache":
                index += 1
                guard index < args.count else { throw CLIError.badValue("--disk-cache") }
                cacheConfiguration.enableDiskCache = try Self.parseBool(args[index], option: "--disk-cache")
            case "--disk-cache-dir":
                index += 1
                guard index < args.count, !args[index].isEmpty else {
                    throw CLIError.badValue("--disk-cache-dir")
                }
                cacheConfiguration.diskCacheDir = URL(fileURLWithPath: args[index])
                    .standardizedFileURL
            case "--kv-cache":
                index += 1
                guard index < args.count else { throw CLIError.badValue("--kv-cache") }
                cacheConfiguration.defaultKVMode = try Self.parseKVMode(args[index])
            case "--router-advice":
                routerAdvice = true
            case "--inspect":
                inspectOnly = true
            case "--print-memory":
                printMemory = true
            case "--json":
                jsonOutput = true
            case "--metrics-jsonl":
                index += 1
                guard index < args.count, !args[index].isEmpty else {
                    throw CLIError.badValue("--metrics-jsonl")
                }
                metricsJSONLURL = URL(fileURLWithPath: args[index]).standardizedFileURL
            case "--activity-gate":
                index += 1
                guard index < args.count,
                    let value = Double(args[index]),
                    (0...100).contains(value)
                else {
                    throw CLIError.badValue("--activity-gate")
                }
                activityGatePercent = value
            case "--activity-gate-report-only", "--no-activity-gate-stop":
                activityGateStopOnFailure = false
            case "--activity-gate-stop":
                activityGateStopOnFailure = true
            case "--file-read-gate-mb-per-token", "--profile-read-gate-mb-per-token":
                index += 1
                guard index < args.count,
                    let value = Double(args[index]),
                    value >= 0
                else {
                    throw CLIError.badValue(arg)
                }
                fileReadGateMBPerGeneratedToken = value
            case "--file-read-gate-report-only", "--profile-read-gate-report-only":
                fileReadGateStopOnFailure = false
            case "--file-read-gate-stop", "--profile-read-gate-stop":
                fileReadGateStopOnFailure = true
            default:
                positional.append(arg)
            }
            index += 1
        }

        let requiredPositionals = inspectOnly ? 1 : (turns.isEmpty ? 2 : 1)
        guard positional.count >= requiredPositionals else {
            throw CLIError.missingArguments
        }
        if inspectOnly, activityGatePercent != nil {
            throw CLIError.activityGateRequiresLoad
        }
        if jsonOutput, !inspectOnly {
            throw CLIError.jsonRequiresInspect
        }
        self.modelDirectory = URL(fileURLWithPath: positional[0]).standardizedFileURL
        if turns.isEmpty {
            self.turns = [positional.dropFirst().joined(separator: " ")]
        } else {
            self.turns = turns
        }
        self.maxTokens = maxTokens
        self.prefillStepSize = prefillStepSize
        self.temperature = temperature
        self.topP = topP
        if !expectedOutputs.isEmpty, expectedOutputs.count != self.turns.count {
            throw CLIError.expectedCountMismatch
        }
        self.expectedOutputs = expectedOutputs
        self.inspectOnly = inspectOnly
        self.printMemory = printMemory
        self.activityGatePercent = activityGatePercent
        self.activityGateStopOnFailure = activityGateStopOnFailure
        self.jsonOutput = jsonOutput
        self.metricsJSONLURL = metricsJSONLURL
        self.enableCompiledDecode = enableCompiledDecode
        self.compiledMaxCacheLength = compiledMaxCacheLength
        self.allowMiniMaxCompiledDecode = allowMiniMaxCompiledDecode
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
        self.minVisibleChars = minVisibleChars
        self.minGenerationTokens = minGenerationTokens
        self.failOnLengthStop = failOnLengthStop
        self.fileReadGateMBPerGeneratedToken = fileReadGateMBPerGeneratedToken
        self.fileReadGateStopOnFailure = fileReadGateStopOnFailure
        self.activeExpertTrace = activeExpertTrace
        let enableActiveExpertStreaming =
            activeExpertStreamingOverride ?? MLXPressLoadConfiguration.default.enableActiveExpertStreaming
        self.loadConfiguration = MLXPressLoadConfiguration(
            compression: compression,
            memoryLimit: .default,
            useMmapSafetensors: useMmapSafetensors,
            enableRouterAdvice: routerAdvice,
            enableActiveExpertStreaming: enableActiveExpertStreaming,
            prestack: prestack,
            cache: cacheConfiguration)
    }

    private static func parseCompression(_ raw: String) throws -> MLXPressCompressionPolicy {
        let value = raw.lowercased()
        if value == "off" || value == "false" || value == "0" {
            return .disabled
        }
        if value == "auto" {
            return .auto(envFallback: true)
        }
        if let pct = Int(value), (0...95).contains(pct) {
            return .enabled(coldFraction: Double(pct) / 100.0)
        }
        throw CLIError.badValue("--mlxpress")
    }

    private static func parseBool(_ raw: String, option: String) throws -> Bool {
        switch raw.lowercased() {
        case "on", "true", "yes", "1":
            return true
        case "off", "false", "no", "0":
            return false
        default:
            throw CLIError.badValue(option)
        }
    }

    private static func parseKVMode(_ raw: String) throws -> KVQuantizationMode {
        switch raw.lowercased() {
        case "none", "off", "false", "0":
            return .none
        case "turboquant", "turbo-quant", "tq", "on", "true", "1":
            return .turboQuant(keyBits: 3, valueBits: 3)
        default:
            throw CLIError.badValue("--kv-cache")
        }
    }
}

private func printBundleFacts(_ facts: MLXPressBundleFacts) {
    print("Bundle: \(facts.directory.path)")
    print("Format: \(facts.format.rawValue)")
    print("Model type: \(facts.modelType ?? "unknown")")
    print("Architectures: \(facts.architecture.architectureNames.isEmpty ? "unknown" : facts.architecture.architectureNames.joined(separator: ", "))")
    print("Attention kinds: \(facts.architecture.attentionSummary)")
    print("Matmul kernels: \(facts.architecture.matmulSummary)")
    print("Position encodings: \(facts.architecture.positionSummary)")
    print("Position vectors: \(facts.architecture.positionVectorSummary)")
    print("Cache storage/encode: \(facts.architecture.cacheStorageSummary)")
    print("Cache encode details: \(facts.architecture.cacheEncodingSummary)")
    print("Companion state: \(facts.architecture.companionStateSummary)")
    print("Hybrid split: \(facts.architecture.hybridSplitSummary)")
    print("Media cache identity: \(facts.architecture.mediaCacheSummary)")
    print("Safetensors: \(MLXPressFormatBytes(facts.totalSafetensorsBytes))")
    print("Routed MoE: \(facts.isRouted ? "yes" : "no")")
    if let experts = facts.numRoutedExperts {
        print("Routed experts: \(experts)")
    }
    if let topK = facts.topK {
        print("Top-k experts: \(topK)")
    }
    print("Tokenizer JSON: \(facts.hasTokenizerJSON ? "yes" : "no")")
    print("Safetensors index: \(facts.hasSafetensorsIndex ? "yes" : "no")")
    print("Auto compression eligible: \(facts.autoCompressionEligible ? "yes" : "no")")
}

private func printReadinessChecklist(_ readiness: MLXPressModelReadinessChecklist) {
    print("Readiness family: \(readiness.family)")
    print("Readiness state: \(readiness.overallState.rawValue)")
    print("Attention/cache architecture: \(readiness.attentionArchitecture)")
    print("Load method: \(readiness.loadMethod)")
    print("Summary: \(readiness.summary)")
    print("Checklist:")
    for item in readiness.items {
        print("- \(item.state.rawValue) \(item.gate): \(item.detail)")
        if let evidence = item.evidence {
            print("  evidence: \(evidence)")
        }
    }
    if !readiness.blockers.isEmpty {
        print("Blockers:")
        for blocker in readiness.blockers {
            print("- \(blocker)")
        }
    }
    print("Required proof:")
    for proof in readiness.requiredProofs {
        print("- \(proof)")
    }
}

private func printInspectJSON(
    facts: MLXPressBundleFacts,
    readiness: MLXPressModelReadinessChecklist,
    memory: MLXPressMemorySnapshot?
) throws {
    let output = InspectJSON(
        bundle: BundleFactsJSON(facts),
        readiness: ReadinessJSON(readiness),
        memory: memory.map(MemorySnapshotJSON.init))
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(output)
    FileHandle.standardOutput.write(data)
    print("")
}

private func printRuntimeCapabilitiesJSONOrText(jsonOutput: Bool) throws {
    let capabilities = MLXPressRuntimeCapabilities.current()
    if jsonOutput {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(RuntimeCapabilitiesJSON(capabilities))
        FileHandle.standardOutput.write(data)
        print("")
    } else {
        print("Canonical mmap routed advisor: \(capabilities.hasCanonicalMmapRoutedAdvisor ? "yes" : "no")")
        print("Canonical mmap expert advisor: \(capabilities.hasCanonicalMmapExpertAdvisor ? "yes" : "no")")
        print("Canonical mmap layer advisor: \(capabilities.hasCanonicalMmapLayerAdvisor ? "yes" : "no")")
        print("Canonical mmap safetensors: \(capabilities.hasCanonicalMmapSafetensorsSupport ? "yes" : "no")")
        print("Activity compression ready: \(capabilities.activityCompressionReady ? "yes" : "no")")
    }
}

private struct RuntimeCapabilitiesJSON: Encodable {
    let hasCanonicalMmapRoutedAdvisor: Bool
    let hasCanonicalMmapExpertAdvisor: Bool
    let hasCanonicalMmapLayerAdvisor: Bool
    let hasCanonicalMmapSafetensorsSupport: Bool
    let activityCompressionReady: Bool

    init(_ capabilities: MLXPressRuntimeCapabilities) {
        self.hasCanonicalMmapRoutedAdvisor = capabilities.hasCanonicalMmapRoutedAdvisor
        self.hasCanonicalMmapExpertAdvisor = capabilities.hasCanonicalMmapExpertAdvisor
        self.hasCanonicalMmapLayerAdvisor = capabilities.hasCanonicalMmapLayerAdvisor
        self.hasCanonicalMmapSafetensorsSupport = capabilities.hasCanonicalMmapSafetensorsSupport
        self.activityCompressionReady = capabilities.activityCompressionReady
    }
}

private struct InspectJSON: Encodable {
    let bundle: BundleFactsJSON
    let readiness: ReadinessJSON
    let memory: MemorySnapshotJSON?
}

private struct ReadinessJSON: Encodable {
    let family: String
    let attentionArchitecture: String
    let loadMethod: String
    let overallState: String
    let summary: String
    let items: [ReadinessItemJSON]
    let requiredProofs: [String]
    let blockers: [String]

    init(_ readiness: MLXPressModelReadinessChecklist) {
        self.family = readiness.family
        self.attentionArchitecture = readiness.attentionArchitecture
        self.loadMethod = readiness.loadMethod
        self.overallState = readiness.overallState.rawValue
        self.summary = readiness.summary
        self.items = readiness.items.map(ReadinessItemJSON.init)
        self.requiredProofs = readiness.requiredProofs
        self.blockers = readiness.blockers
    }
}

private struct ReadinessItemJSON: Encodable {
    let gate: String
    let state: String
    let detail: String
    let evidence: String?

    init(_ item: MLXPressReadinessItem) {
        self.gate = item.gate
        self.state = item.state.rawValue
        self.detail = item.detail
        self.evidence = item.evidence
    }
}

private struct BundleFactsJSON: Encodable {
    let directory: String
    let format: String
    let modelType: String?
    let architecture: MLXPressArchitectureFacts
    let totalSafetensorsBytes: UInt64
    let isRouted: Bool
    let numRoutedExperts: Int?
    let topK: Int?
    let hasTokenizerJSON: Bool
    let hasSafetensorsIndex: Bool
    let physicalMemoryBytes: UInt64
    let autoCompressionEligible: Bool

    init(_ facts: MLXPressBundleFacts) {
        self.directory = facts.directory.path
        self.format = facts.format.rawValue
        self.modelType = facts.modelType
        self.architecture = facts.architecture
        self.totalSafetensorsBytes = facts.totalSafetensorsBytes
        self.isRouted = facts.isRouted
        self.numRoutedExperts = facts.numRoutedExperts
        self.topK = facts.topK
        self.hasTokenizerJSON = facts.hasTokenizerJSON
        self.hasSafetensorsIndex = facts.hasSafetensorsIndex
        self.physicalMemoryBytes = facts.physicalMemoryBytes
        self.autoCompressionEligible = facts.autoCompressionEligible
    }
}

private struct MemorySnapshotJSON: Encodable {
    let residentSizeBytes: UInt64
    let physicalFootprintBytes: UInt64?
    let physicalMemoryBytes: UInt64
    let mlxActiveMemoryBytes: UInt64
    let mlxCacheMemoryBytes: UInt64
    let mlxPeakMemoryBytes: UInt64

    init(_ snapshot: MLXPressMemorySnapshot) {
        self.residentSizeBytes = snapshot.residentSizeBytes
        self.physicalFootprintBytes = snapshot.physicalFootprintBytes
        self.physicalMemoryBytes = snapshot.physicalMemoryBytes
        self.mlxActiveMemoryBytes = snapshot.mlxActiveMemoryBytes
        self.mlxCacheMemoryBytes = snapshot.mlxCacheMemoryBytes
        self.mlxPeakMemoryBytes = snapshot.mlxPeakMemoryBytes
    }
}

private func printMemorySnapshot(_ snapshot: MLXPressMemorySnapshot, label: String) {
    var parts = [
        "RSS=\(MLXPressFormatBytes(snapshot.residentSizeBytes))",
        "mlx-active=\(MLXPressFormatBytes(snapshot.mlxActiveMemoryBytes))",
        "mlx-cache=\(MLXPressFormatBytes(snapshot.mlxCacheMemoryBytes))",
        "mlx-peak=\(MLXPressFormatBytes(snapshot.mlxPeakMemoryBytes))",
        "physical=\(MLXPressFormatBytes(snapshot.physicalMemoryBytes))",
    ]
    if let footprint = snapshot.physicalFootprintBytes {
        parts.insert(
            "footprint=\(MLXPressFormatBytes(footprint))",
            at: 1)
    }
    fputs("\(label): \(parts.joined(separator: " "))\n", stderr)
}

private func printMemoryDelta(
    preLoad: MLXPressMemorySnapshot,
    postLoad: MLXPressMemorySnapshot,
    modelBytes: UInt64,
    label: String
) {
    var parts = [
        "RSS-increase=\(formatSignedBytes(signedDelta(postLoad.residentSizeBytes, preLoad.residentSizeBytes)))",
        "mlx-active-increase=\(formatSignedBytes(signedDelta(postLoad.mlxActiveMemoryBytes, preLoad.mlxActiveMemoryBytes)))",
        "mlx-cache-increase=\(formatSignedBytes(signedDelta(postLoad.mlxCacheMemoryBytes, preLoad.mlxCacheMemoryBytes)))",
        "mlx-peak-increase=\(formatSignedBytes(signedDelta(postLoad.mlxPeakMemoryBytes, preLoad.mlxPeakMemoryBytes)))",
    ]
    if let before = preLoad.physicalFootprintBytes,
       let after = postLoad.physicalFootprintBytes
    {
        let delta = signedDelta(after, before)
        parts.append("footprint-increase=\(formatSignedBytes(delta))")
        if modelBytes > 0 {
            let ratio = Double(max(delta, 0)) / Double(modelBytes) * 100.0
            parts.append("model-bytes=\(MLXPressFormatBytes(modelBytes))")
            parts.append("ratio=\(formatPercent(ratio))")
        }
    }
    fputs("\(label): \(parts.joined(separator: " "))\n", stderr)
}

private func printActivityCompressionCheck(
    _ check: MLXPressActivityCompressionCheck,
    label: String
) {
    let increase = check.footprintIncreaseBytes.map(MLXPressFormatBytes) ?? "unavailable"
    let increasePercent = check.footprintIncreasePercent.map(formatPercent) ?? "unavailable"
    let parts = [
        "verdict=\(check.verdict.rawValue)",
        "footprint-increase=\(increase)",
        "model-bytes=\(MLXPressFormatBytes(check.modelBytes))",
        "ratio=\(increasePercent)",
        "allowed=\(MLXPressFormatBytes(check.maxAllowedFootprintIncreaseBytes))",
        "gate=\(formatPercent(check.maxFootprintPercent))",
    ]
    fputs("\(label): \(parts.joined(separator: " "))\n", stderr)
}

private func printGenerationTelemetry(_ info: GenerateCompletionInfo, turn: Int) {
    let parts = [
        "turn=\(turn)",
        "prompt-tokens=\(info.promptTokenCount)",
        "prompt-tokens/s=\(formatDecimal(info.promptTokensPerSecond))",
        "generation-tokens=\(info.generationTokenCount)",
        "tokens/s=\(formatDecimal(info.tokensPerSecond))",
        "prompt-time=\(formatSeconds(info.promptTime))",
        "generation-time=\(formatSeconds(info.generateTime))",
        "stop=\(String(describing: info.stopReason))",
        "unclosed-reasoning=\(info.unclosedReasoning)",
    ]
    fputs("Generation telemetry \(parts.joined(separator: " "))\n", stderr)
}

private struct TurnCoherencyReport {
    var passed: Bool
    var visibleStatus: String
    var anyStatus: String
    var printableStatus: String
    var expectedStatus: String
    var loopStatus: String
    var loopReason: String
    var visibleLoopStatus: String
    var reasoningLoopStatus: String
    var minVisibleStatus: String
    var minGeneratedStatus: String
    var lengthStopStatus: String
    var visibleChars: Int
    var reasoningChars: Int
    var visiblePreview: String
    var reasoningPreview: String

    var jsonObject: [String: Any] {
        [
            "passed": passed,
            "visible": visibleStatus,
            "any": anyStatus,
            "printable": printableStatus,
            "expected": expectedStatus,
            "loop": loopStatus,
            "loop_reason": loopReason,
            "visible_loop": visibleLoopStatus,
            "reasoning_loop": reasoningLoopStatus,
            "min_visible": minVisibleStatus,
            "min_generated": minGeneratedStatus,
            "length_stop": lengthStopStatus,
            "visible_chars": visibleChars,
            "reasoning_chars": reasoningChars,
            "visible_preview": visiblePreview,
            "reasoning_preview": reasoningPreview,
        ]
    }
}

private func printTurnCoherency(
    visibleText: String,
    reasoningText: String,
    completionInfo: GenerateCompletionInfo?,
    expectedText: String?,
    turn: Int,
    minVisibleChars: Int,
    minGenerationTokens: Int,
    failOnLengthStop: Bool
) -> TurnCoherencyReport {
    let trimmedVisible = visibleText.trimmingCharacters(in: .whitespacesAndNewlines)
    let trimmedReasoning = reasoningText.trimmingCharacters(in: .whitespacesAndNewlines)
    let combined = [trimmedReasoning, trimmedVisible]
        .filter { !$0.isEmpty }
        .joined(separator: "\n")
    let hasInvalidControl = combined.unicodeScalars.contains { scalar in
        CharacterSet.controlCharacters.contains(scalar)
            && scalar != "\n"
            && scalar != "\r"
            && scalar != "\t"
    }
    let expectedStatus: String
    if let expectedText {
        expectedStatus = trimmedVisible
            .range(of: expectedText, options: [.caseInsensitive, .diacriticInsensitive]) == nil
            ? "fail"
            : "pass"
    } else {
        expectedStatus = "n/a"
    }
    let visibleLoopAssessment = assessLooping(in: trimmedVisible)
    let reasoningLoopAssessment = assessLooping(in: trimmedReasoning)
    let combinedLoopAssessment = combineLoopAssessments(
        visible: visibleLoopAssessment,
        reasoning: reasoningLoopAssessment)
    let minVisibleStatus = minVisibleChars > 0
        ? (trimmedVisible.count >= minVisibleChars ? "pass" : "fail")
        : "n/a"
    let minGeneratedStatus: String
    if minGenerationTokens > 0 {
        let generated = completionInfo?.generationTokenCount ?? 0
        minGeneratedStatus = generated >= minGenerationTokens ? "pass" : "fail"
    } else {
        minGeneratedStatus = "n/a"
    }
    let stopReason = completionInfo.map { String(describing: $0.stopReason) } ?? "missing"
    let lengthStopStatus = failOnLengthStop
        ? (stopReason == "length" ? "fail" : "pass")
        : "n/a"
    let visibleStatus = trimmedVisible.isEmpty ? "fail" : "pass"
    let anyStatus = combined.isEmpty ? "fail" : "pass"
    let printableStatus = hasInvalidControl ? "fail" : "pass"
    let visiblePreview = percentEncodedPreview(trimmedVisible)
    let reasoningPreview = percentEncodedPreview(trimmedReasoning)
    let parts = [
        "turn=\(turn)",
        "visible=\(visibleStatus)",
        "any=\(anyStatus)",
        "printable=\(printableStatus)",
        "expected=\(expectedStatus)",
        "loop=\(combinedLoopAssessment.status)",
        "loop-reason=\(combinedLoopAssessment.reason)",
        "visible-loop=\(visibleLoopAssessment.status)",
        "reasoning-loop=\(reasoningLoopAssessment.status)",
        "min-visible=\(minVisibleStatus)",
        "min-generated=\(minGeneratedStatus)",
        "length-stop=\(lengthStopStatus)",
        "visible-chars=\(trimmedVisible.count)",
        "reasoning-chars=\(trimmedReasoning.count)",
        "preview=\(visiblePreview)",
        "reasoning-preview=\(reasoningPreview)",
    ]
    fputs("Turn coherency \(parts.joined(separator: " "))\n", stderr)
    let passed = !trimmedVisible.isEmpty
        && !combined.isEmpty
        && !hasInvalidControl
        && (expectedStatus == "pass" || expectedStatus == "n/a")
        && combinedLoopAssessment.status == "pass"
        && (minVisibleStatus == "pass" || minVisibleStatus == "n/a")
        && (minGeneratedStatus == "pass" || minGeneratedStatus == "n/a")
        && (lengthStopStatus == "pass" || lengthStopStatus == "n/a")
    return TurnCoherencyReport(
        passed: passed,
        visibleStatus: visibleStatus,
        anyStatus: anyStatus,
        printableStatus: printableStatus,
        expectedStatus: expectedStatus,
        loopStatus: combinedLoopAssessment.status,
        loopReason: combinedLoopAssessment.reason,
        visibleLoopStatus: visibleLoopAssessment.status,
        reasoningLoopStatus: reasoningLoopAssessment.status,
        minVisibleStatus: minVisibleStatus,
        minGeneratedStatus: minGeneratedStatus,
        lengthStopStatus: lengthStopStatus,
        visibleChars: trimmedVisible.count,
        reasoningChars: trimmedReasoning.count,
        visiblePreview: visiblePreview,
        reasoningPreview: reasoningPreview)
}

private struct LoopAssessment {
    var status: String
    var reason: String
}

private func combineLoopAssessments(
    visible: LoopAssessment,
    reasoning: LoopAssessment
) -> LoopAssessment {
    if visible.status == "fail" {
        return LoopAssessment(status: "fail", reason: "visible-\(visible.reason)")
    }
    if reasoning.status == "fail" {
        return LoopAssessment(status: "fail", reason: "reasoning-\(reasoning.reason)")
    }
    if visible.reason == "empty", reasoning.reason == "empty" {
        return LoopAssessment(status: "pass", reason: "empty")
    }
    return LoopAssessment(status: "pass", reason: "none")
}

private func assessLooping(in text: String) -> LoopAssessment {
    guard !text.isEmpty else {
        return LoopAssessment(status: "pass", reason: "empty")
    }

    var previousScalar: UnicodeScalar?
    var scalarRun = 0
    for scalar in text.unicodeScalars {
        if CharacterSet.whitespacesAndNewlines.contains(scalar) {
            previousScalar = nil
            scalarRun = 0
            continue
        }
        if scalar == previousScalar {
            scalarRun += 1
        } else {
            previousScalar = scalar
            scalarRun = 1
        }
        if scalarRun >= 24 {
            return LoopAssessment(status: "fail", reason: "char-run")
        }
    }

    let lines = text
        .components(separatedBy: .newlines)
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
        .filter { !$0.isEmpty }
    if lines.count >= 3 {
        var previousLine = ""
        var lineRun = 0
        for line in lines {
            if line == previousLine {
                lineRun += 1
            } else {
                previousLine = line
                lineRun = 1
            }
            if line.count >= 4, lineRun >= 3 {
                return LoopAssessment(status: "fail", reason: "line-run")
            }
        }
        if lines.count >= 8, Set(lines).count <= 2 {
            return LoopAssessment(status: "fail", reason: "low-line-variety")
        }
    }

    let words = text
        .lowercased()
        .split { character in
            !character.isLetter && !character.isNumber
        }
        .map(String.init)
    if words.count >= 6 {
        var previousWord = ""
        var wordRun = 0
        for word in words {
            if word == previousWord {
                wordRun += 1
            } else {
                previousWord = word
                wordRun = 1
            }
            if wordRun >= 6 {
                return LoopAssessment(status: "fail", reason: "word-run")
            }
        }
    }

    if words.count >= 8 {
        let maxGram = min(8, words.count / 4)
        if maxGram >= 2 {
            for gramSize in 2...maxGram {
                var index = 0
                while index + gramSize * 4 <= words.count {
                    let gram = Array(words[index..<(index + gramSize)])
                    var repeats = 1
                    var next = index + gramSize
                    while next + gramSize <= words.count,
                          Array(words[next..<(next + gramSize)]) == gram
                    {
                        repeats += 1
                        next += gramSize
                    }
                    if repeats >= 4 {
                        return LoopAssessment(status: "fail", reason: "ngram-run")
                    }
                    index += max(1, repeats * gramSize)
                }
            }
        }
    }

    return LoopAssessment(status: "pass", reason: "none")
}

private func percentEncodedPreview(_ text: String) -> String {
    let prefix = String(text.prefix(96))
    var allowed = CharacterSet.alphanumerics
    allowed.insert(charactersIn: "-_.,:;!?/()[]")
    return prefix.addingPercentEncoding(withAllowedCharacters: allowed) ?? ""
}

private func isoTimestamp() -> String {
    ISO8601DateFormatter().string(from: Date())
}

private func jsonNumber(_ value: UInt64) -> NSNumber {
    NSNumber(value: value <= UInt64(Int64.max) ? Int64(value) : Int64.max)
}

private func jsonOptionalNumber(_ value: UInt64?) -> Any {
    guard let value else { return NSNull() }
    return jsonNumber(value)
}

private func jsonDouble(_ value: Double) -> Any {
    value.isFinite ? value : NSNull()
}

private func jsonOptionalDouble(_ value: Double?) -> Any {
    guard let value else { return NSNull() }
    return jsonDouble(value)
}

private func jsonFloat(_ value: Float) -> Any {
    value.isFinite ? Double(value) : NSNull()
}

private func formatDecimal(_ value: Double) -> String {
    String(format: "%.2f", value.isFinite ? value : 0)
}

private func formatSeconds(_ value: Double) -> String {
    String(format: "%.3fs", value.isFinite ? value : 0)
}

private func signedDelta(_ after: UInt64, _ before: UInt64) -> Int64 {
    if after >= before {
        return Int64(after - before)
    }
    return -Int64(before - after)
}

private func counterDelta(_ after: UInt64, _ before: UInt64) -> UInt64 {
    after >= before ? after - before : 0
}

private func formatSignedBytes(_ value: Int64) -> String {
    if value < 0 {
        return "-\(MLXPressFormatBytes(UInt64(-value)))"
    }
    return MLXPressFormatBytes(UInt64(value))
}

private func formatPercent(_ value: Double) -> String {
    String(format: "%.1f%%", value)
}

private enum CLIError: Error, CustomStringConvertible {
    case missingArguments
    case badValue(String)
    case activityGateRequiresLoad
    case loadRequiresLocalWeights
    case jsonRequiresInspect
    case expectedCountMismatch
    case invalidMetricsObject

    var description: String {
        switch self {
        case .missingArguments:
            return "missing required model directory and prompt"
        case .badValue(let option):
            return "invalid value for \(option)"
        case .activityGateRequiresLoad:
            return "--activity-gate requires a real model load; it cannot run with --inspect"
        case .loadRequiresLocalWeights:
            return "refusing to load a model directory with no top-level local safetensors; use --inspect to audit bundle metadata first"
        case .jsonRequiresInspect:
            return "--json is currently supported only with --inspect"
        case .expectedCountMismatch:
            return "--expect count must match the number of turns"
        case .invalidMetricsObject:
            return "internal error: metrics JSONL record is not valid JSON"
        }
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
