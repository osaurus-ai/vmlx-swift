/// Explicit scope for the actual Flash target-verification entry point.
/// Ordinary prefill and other model families never establish this scope.
public enum FlashVerificationScope {
    @TaskLocal private static var rows = 0

    public static func withVerification<Result>(
        inputShape: [Int], operation: () throws -> Result
    ) rethrows -> Result {
        let admittedRows = inputShape.count == 2 && inputShape[0] == 1
            && (2...8).contains(inputShape[1]) ? inputShape[1] : 0
        return try $rows.withValue(admittedRows, operation: operation)
    }

    /// Whether this tensor belongs to the active small-row Flash target verification.
    /// Does not establish scope or admit ordinary prefill, AR, or draft-head calls.
    public static func usesRowExactVerification(inputShape: [Int]) -> Bool {
        rows > 0 && inputShape.count == 3 && inputShape[0] == 1
            && inputShape[1] == rows && inputShape[2] > 0
    }

    static func usesMappedDecode(inputShape: [Int], routes: Int) -> Bool {
        // S1 itself uses prefill when routes >= 64. Preserve that arithmetic.
        rows > 0 && inputShape.count == 3 && inputShape[0] == 1
            && inputShape[1] == rows && routes > 0 && routes < 64
    }
}
