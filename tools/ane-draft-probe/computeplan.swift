// Per-op Neural Engine placement report for a compiled Core ML program.
//
// Walks the MIL program through MLComputePlan and prints, for every op, the
// devices that support it and the one Core ML would pick. This is how a
// "does this op run on the ANE" question gets answered by the framework
// instead of by guessing from a bare ANECCompile() FAILED.
//
// Build: swiftc -O -framework CoreML tools/ane-draft-probe/computeplan.swift -o /tmp/computeplan
// Run:   /tmp/computeplan /path/to/head.mlmodelc [--all]

import CoreML
import Foundation

@available(macOS 14.4, *)
func deviceName(_ d: MLComputeDevice) -> String {
    switch d {
    case .cpu: return "CPU"
    case .gpu: return "GPU"
    case .neuralEngine: return "ANE"
    @unknown default: return "?"
    }
}

@available(macOS 14.4, *)
func run() async throws {
    let args = CommandLine.arguments
    guard args.count >= 2 else {
        print("usage: computeplan <model.mlmodelc> [--all]")
        exit(2)
    }
    let url = URL(fileURLWithPath: args[1])
    let showAll = args.contains("--all")
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
    guard case .program(let program) = plan.modelStructure,
          let main = program.functions["main"] else {
        print("not an ML program with a main function")
        exit(1)
    }
    var total = 0, onANE = 0, notANE: [String] = []
    for op in main.block.operations {
        total += 1
        let usage = plan.deviceUsage(for: op)
        let supported = usage?.supported.map(deviceName).joined(separator: ",") ?? "-"
        let preferred = usage.map { deviceName($0.preferred) } ?? "-"
        let aneOK = usage?.supported.contains(where: { if case .neuralEngine = $0 { return true }; return false }) ?? false
        if aneOK { onANE += 1 } else { notANE.append("\(op.operatorName) \(op.outputs.first?.name ?? "")  supported=[\(supported)]") }
        if showAll {
            print("\(op.operatorName.padding(toLength: 34, withPad: " ", startingAt: 0)) \(op.outputs.first?.name ?? "") -> preferred=\(preferred) supported=[\(supported)]")
        }
    }
    print("ops: \(total)  ANE-supported: \(onANE)  not-ANE: \(notANE.count)")
    for line in notANE.prefix(40) { print("  NOT-ANE: \(line)") }
}

if #available(macOS 14.4, *) {
    let sem = DispatchSemaphore(value: 0)
    Task {
        do { try await run() } catch { print("error: \(error)"); exit(1) }
        sem.signal()
    }
    sem.wait()
} else {
    print("needs macOS 14.4+")
}
