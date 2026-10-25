import Foundation
import Testing

@testable import MLXLMCommon

struct RuntimeDiagnosticsCaptureTests {
    @Test func exactContentReturnValueAndEmptyWrites() {
        let (value, text) = DiskCacheAccountingTestSupport.capturingRuntimeDiagnostics {
            RuntimeDiagnostics.write(Data())
            RuntimeDiagnostics.write(Data("diagnostic α\n".utf8))
            RuntimeDiagnostics.write(Data("second\n".utf8))
            return 42
        }
        #expect(value == 42)
        #expect(text == "diagnostic α\nsecond\n")
    }

    @Test func nestedScopesRestoreTheirParent() {
        let (_, outer) = DiskCacheAccountingTestSupport.capturingRuntimeDiagnostics {
            RuntimeDiagnostics.write(Data("before\n".utf8))
            let (_, inner) = DiskCacheAccountingTestSupport.capturingRuntimeDiagnostics {
                RuntimeDiagnostics.write(Data("inner\n".utf8))
            }
            #expect(inner == "inner\n")
            RuntimeDiagnostics.write(Data("after\n".utf8))
        }
        #expect(outer == "before\nafter\n")
        #expect(RuntimeDiagnostics.observerForTesting == nil)
    }

    @Test func throwingScopeRestoresItsParent() {
        enum Expected: Error { case failure }
        let (_, outer) = DiskCacheAccountingTestSupport.capturingRuntimeDiagnostics {
            do {
                _ = try DiskCacheAccountingTestSupport.capturingRuntimeDiagnostics {
                    RuntimeDiagnostics.write(Data("inner\n".utf8))
                    throw Expected.failure
                }
                Issue.record("Throwing capture returned normally")
            } catch Expected.failure {
                RuntimeDiagnostics.write(Data("restored\n".utf8))
            } catch {
                Issue.record("Unexpected capture error: \(error)")
            }
        }
        #expect(outer == "restored\n")
        #expect(RuntimeDiagnostics.observerForTesting == nil)
    }

    @Test func simultaneousTasksDoNotCaptureEachOthersMessages() async {
        let results = await withTaskGroup(of: (Int, String).self) { group in
            for index in 0..<32 {
                group.addTask {
                    let (_, text) = DiskCacheAccountingTestSupport.capturingRuntimeDiagnostics {
                        for _ in 0..<64 {
                            RuntimeDiagnostics.write(Data("capture-\(index)\n".utf8))
                        }
                    }
                    return (index, text)
                }
            }
            var results: [(Int, String)] = []
            for await result in group { results.append(result) }
            return results
        }
        #expect(results.count == 32)
        for (index, text) in results {
            #expect(text == String(repeating: "capture-\(index)\n", count: 64))
        }
        #expect(RuntimeDiagnostics.observerForTesting == nil)
    }

    @Test func detachedWorkerExplicitlyForwardsTheScopedObserver() {
        let (_, text) = DiskCacheAccountingTestSupport.capturingRuntimeDiagnostics {
            let observer = RuntimeDiagnostics.observerForTesting
            let done = DispatchSemaphore(value: 0)
            Thread.detachNewThread {
                RuntimeDiagnostics.$observerForTesting.withValue(observer) {
                    RuntimeDiagnostics.write(Data("detached\n".utf8))
                }
                done.signal()
            }
            #expect(done.wait(timeout: .now() + 5) == .success)
        }
        #expect(text == "detached\n")
        #expect(RuntimeDiagnostics.observerForTesting == nil)
    }

    #if canImport(os)
        @Test func observerCannotChangeTheCallersErrno() {
            let saved = errno
            defer { errno = saved }
            errno = E2BIG
            RuntimeDiagnostics.$observerForTesting.withValue({ _ in errno = ERANGE }) {
                RuntimeDiagnostics.write(Data("errno preservation\n".utf8))
            }
            #expect(errno == E2BIG)
        }
    #endif

}
