/// Keeps small routed target-verification batches on the single-token expert
/// reduction path. Prefill, draft-head calls, and ordinary decoding do not
/// establish this scope. It is separate from Flash's other exact-row policies.
public enum RoutedExpertVerificationScope {
    @TaskLocal private static var rows = 0

    public static func withVerification<Result>(
        inputShape: [Int], operation: () throws -> Result
    ) rethrows -> Result {
        let admitted = inputShape.count == 2 && inputShape[0] == 1
            && (2 ... 8).contains(inputShape[1]) ? inputShape[1] : 0
        return try $rows.withValue(admitted, operation: operation)
    }

    static func usesMappedDecode(inputShape: [Int], routes: Int) -> Bool {
        // Preserve the existing S1 choice when a model selects >=64 experts.
        rows > 0 && inputShape.count == 3 && inputShape[0] == 1
            && inputShape[1] == rows && inputShape[2] > 0
            && routes > 0 && routes < 64
    }
}
