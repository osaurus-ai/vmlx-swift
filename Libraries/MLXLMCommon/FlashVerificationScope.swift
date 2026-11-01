import Foundation

/// Explicit scope for the actual Flash target-verification entry point.
/// Ordinary prefill and other model families never establish this scope.
public enum FlashVerificationScope {
    @TaskLocal private static var rows = 0
    /// Verify windows checked row-exactly (every row bit-identical to the AR step): 2...8 rows, i.e. draft depth
    /// 1...7. This is the native-MTP verifier's correctness window (`NativeMTPDepthPolicy.verifierTopRung`).
    public static let rowExactVerifyRows: ClosedRange<Int> = 2 ... 8
    private static let diagnosticDisabled =
        ProcessInfo.processInfo.environment["VMLX_FLASH_ROW_EXACT"] == "0"
    /// Diagnostic only: VMLX_FLASH_ROW_EXACT_OFF=q8,hcjoin,hcdown,hcup,attn,early,router,gdn,moe turns single
    /// exact-row routes off so their verify cost can be attributed. Greedy output may then differ from AR.
    private static let diagnosticOffSites: Set<String> = Set(
        (ProcessInfo.processInfo.environment["VMLX_FLASH_ROW_EXACT_OFF"] ?? "")
            .split(separator: ",").map { String($0) })

    /// A tree is already bounded by its validated drafter plan. This scope does not alter its width.
    public static func withTreeVerification<Result>(
        inputShape: [Int], operation: () throws -> Result
    ) rethrows -> Result {
        let admitted = !diagnosticDisabled && inputShape.count == 2 && inputShape[0] == 1
            && inputShape[1] > 1 ? inputShape[1] : 0
        return try $rows.withValue(admitted, operation: operation)
    }

    public static func withVerification<Result>(
        inputShape: [Int], operation: () throws -> Result
    ) rethrows -> Result {
        // Diagnostic only (measurement of the exact-verify machinery's cost): VMLX_FLASH_ROW_EXACT=0 runs
        // verification without the row-exact routes. Greedy output may then differ from AR.
        let admittedRows =
            diagnosticDisabled
            ? 0
            : (inputShape.count == 2 && inputShape[0] == 1
                && rowExactVerifyRows.contains(inputShape[1]) ? inputShape[1] : 0)
        return try $rows.withValue(admittedRows, operation: operation)
    }

    /// Whether this tensor belongs to the active small-row Flash target verification.
    /// Does not establish scope or admit ordinary prefill, AR, or draft-head calls.
    public static func usesRowExactVerification(inputShape: [Int], site: String = "") -> Bool {
        if !site.isEmpty, diagnosticOffSites.contains(site) { return false }
        return rows > 0 && inputShape.count == 3 && inputShape[0] == 1
            && inputShape[1] == rows && inputShape[2] > 0
    }

    static func usesMappedDecode(inputShape: [Int], routes: Int) -> Bool {
        // S1 itself uses prefill when routes >= 64. Preserve that arithmetic.
        rows > 0 && inputShape.count == 3 && inputShape[0] == 1
            && inputShape[1] == rows && routes > 0 && routes < 64
    }
}
