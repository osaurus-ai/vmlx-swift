// Branch-only: records every MSA path pick (MiniMaxM3MSA.hooks, perf/minimax-m3-block-sparse-msa).
import Foundation
@testable import MLXLLM

final class PathSpy: @unchecked Sendable {
    private let lock = NSLock()
    private var picks: [String] = []
    static let shared = PathSpy()
    var available: Bool { true }
    func install() {
        MiniMaxM3MSA.hooks.pathObserver = { [weak self] path in
            guard let self else { return }
            self.lock.withLock { self.picks.append(path.rawValue) }
        }
    }
    func drain() -> [String] { lock.withLock { defer { picks = [] }; return picks } }
}

func msaDiagnostics() -> String {
    let err = MiniMaxM3MSA.matrixUnitSelfCheck()
    return "matrixUnitSelfCheck err \(err), matrixUnitsAvailable \(MiniMaxM3MSA.matrixUnitsAvailable())"
}
