import Darwin
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

private final class NullTokenizerLoader: TokenizerLoader, @unchecked Sendable {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        NullTokenizer()
    }
}

private final class NullTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private struct Options {
    var modelPath: String?
    var jobs = 2
    var staggerMilliseconds = 0
    var holdSeconds = 0

    init(arguments: [String]) throws {
        var index = 0
        while index < arguments.count {
            let arg = arguments[index]
            switch arg {
            case "--model":
                index += 1
                guard index < arguments.count else { throw usage("missing value for --model") }
                modelPath = arguments[index]
            case "--jobs":
                index += 1
                guard index < arguments.count, let value = Int(arguments[index]), value > 0 else {
                    throw usage("invalid value for --jobs")
                }
                jobs = value
            case "--stagger-ms":
                index += 1
                guard index < arguments.count, let value = Int(arguments[index]), value >= 0 else {
                    throw usage("invalid value for --stagger-ms")
                }
                staggerMilliseconds = value
            case "--hold-seconds":
                index += 1
                guard index < arguments.count, let value = Int(arguments[index]), value >= 0 else {
                    throw usage("invalid value for --hold-seconds")
                }
                holdSeconds = value
            case "--help", "-h":
                throw usage(nil)
            default:
                throw usage("unknown argument \(arg)")
            }
            index += 1
        }

        guard modelPath != nil else { throw usage("missing --model") }
    }

    private func usage(_ reason: String?) -> NSError {
        let prefix = reason.map { "\($0)\n\n" } ?? ""
        return NSError(
            domain: "LoadRaceProbe",
            code: 64,
            userInfo: [
                NSLocalizedDescriptionKey:
                    prefix + """
                    usage: LoadRaceProbe --model PATH [--jobs 2] [--stagger-ms 0] [--hold-seconds 0]

                    Starts multiple loadModel tasks in one process to reproduce duplicate-load
                    materialization races such as Sentry APPLE-MACOS-25/31/5M.
                    """,
            ])
    }
}

@main
struct LoadRaceProbe {
    static func main() async {
        do {
            let options = try Options(arguments: Array(CommandLine.arguments.dropFirst()))
            let modelURL = URL(fileURLWithPath: options.modelPath!).resolvingSymlinksInPath()
            guard FileManager.default.fileExists(atPath: modelURL.path) else {
                throw NSError(
                    domain: "LoadRaceProbe",
                    code: 66,
                    userInfo: [NSLocalizedDescriptionKey: "model path does not exist: \(modelURL.path)"])
            }

            print("LOAD_RACE_BEGIN model=\(modelURL.path) jobs=\(options.jobs) staggerMs=\(options.staggerMilliseconds)")
            let startedAt = CFAbsoluteTimeGetCurrent()

            try await withThrowingTaskGroup(of: String.self) { group in
                for job in 0..<options.jobs {
                    group.addTask {
                        if options.staggerMilliseconds > 0, job > 0 {
                            try await Task.sleep(
                                nanoseconds: UInt64(options.staggerMilliseconds * job) * 1_000_000)
                        }
                        let start = CFAbsoluteTimeGetCurrent()
                        print("LOAD_JOB_BEGIN job=\(job)")
                        let context = try await MLXLMCommon.loadModel(
                            from: modelURL,
                            using: NullTokenizerLoader())
                        MLX.eval(context.model)
                        if options.holdSeconds > 0 {
                            try await Task.sleep(
                                nanoseconds: UInt64(options.holdSeconds) * 1_000_000_000)
                        }
                        let elapsed = CFAbsoluteTimeGetCurrent() - start
                        return "LOAD_JOB_OK job=\(job) elapsed=\(String(format: "%.3f", elapsed)) modelType=\(type(of: context.model))"
                    }
                }

                for try await line in group {
                    print(line)
                }
            }

            print("LOAD_RACE_OK elapsed=\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - startedAt))")
        } catch {
            fputs("LOAD_RACE_FAIL \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}
