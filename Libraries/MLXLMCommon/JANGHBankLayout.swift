import Cmlx
import MLX

/// Metadata-only admission for model-sized banks. Never evaluates or repacks an
/// input: lazy arrays must be prepared explicitly by their storage owner first.
enum JANGHBankLayout {
    static func requireReadyRowContiguous(_ bank: MLXArray, role: String) throws {
        var available = false
        guard _mlx_array_is_available(&available, bank.ctx) == 0, available else {
            throw JANGHFormatContract.ValidationError.invalid(
                "JANGH \(role) bank is unavailable; implicit evaluation is disabled")
        }
        // Layout flags are stable only after availability. Do not replace this
        // with pre-evaluation strides or asData(), which evaluates the bank.
        var rowContiguous = false
        guard _mlx_array_is_row_contiguous(&rowContiguous, bank.ctx) == 0, rowContiguous else {
            throw JANGHFormatContract.ValidationError.invalid(
                "JANGH \(role) bank is not row contiguous; implicit bank copies are disabled")
        }
    }
}
