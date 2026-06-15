import XCTest
@testable import vMLXFlux
@testable import vMLXFluxKit
@testable import vMLXFluxVideo

/// Pure-Swift shape + math tests. We can't run tests that touch MLX
/// ops because the test binary doesn't have `default.metallib` in its
/// resource bundle, so any Metal op (including the CPU stream's
/// initialization) crashes with "Failed to load the default metallib".
///
/// The MLX-dependent runtime fixtures that DO validate module forward
/// passes live in the running app instead — trigger them via the
/// Terminal tab with a scripted test prompt. See docs/runtime-shape-check.md
/// for the manual validation recipe.
final class ShapeTests: XCTestCase {

    // MARK: - Flow match scheduler (pure Swift)

    func testFlowMatchSigmasDecreaseMonotonically() {
        let scheduler = FlowMatchEulerScheduler(steps: 10, imageSeqLen: 256)
        XCTAssertEqual(scheduler.sigmas.count, 11)
        for i in 0..<10 {
            XCTAssertGreaterThanOrEqual(
                scheduler.sigmas[i],
                scheduler.sigmas[i + 1],
                "sigmas must decrease: \(scheduler.sigmas[i]) → \(scheduler.sigmas[i + 1])"
            )
        }
        // First ≈ 1, last = 0.
        XCTAssertGreaterThan(scheduler.sigmas[0], 0.5)
        XCTAssertEqual(scheduler.sigmas[10], 0, accuracy: 1e-6)
    }

    func testFlowMatchTimestepsMatchSigmas() {
        let scheduler = FlowMatchEulerScheduler(steps: 4, imageSeqLen: 4096)
        XCTAssertEqual(scheduler.timesteps.count, 4,
            "timesteps length = steps (not steps+1)")
        // Each timestep = sigma * 1000.
        for i in 0..<4 {
            XCTAssertEqual(
                scheduler.timesteps[i],
                scheduler.sigmas[i] * 1000.0,
                accuracy: 1e-6
            )
        }
    }

    func testSchedulerShiftIsResolutionDependent() {
        let small = FlowMatchEulerScheduler.computeShift(imageSeqLen: 256)
        let large = FlowMatchEulerScheduler.computeShift(imageSeqLen: 4096)
        XCTAssertLessThan(small, large,
            "large images should get a larger shift")
        // baseShift = 0.5 at min, maxShift = 1.15 at max.
        XCTAssertEqual(small, 0.5, accuracy: 1e-6)
        XCTAssertEqual(large, 1.15, accuracy: 1e-6)
    }

    func testSchedulerShiftClampsAtEndpoints() {
        let below = FlowMatchEulerScheduler.computeShift(imageSeqLen: 100)
        let above = FlowMatchEulerScheduler.computeShift(imageSeqLen: 10000)
        XCTAssertEqual(below, 0.5, accuracy: 1e-6, "clamps at min")
        XCTAssertEqual(above, 1.15, accuracy: 1e-6, "clamps at max")
    }

    // MARK: - FluxDiTConfig presets

    func testFluxDiTConfigPresetsSanity() {
        XCTAssertEqual(FluxDiTConfig.schnell.numDoubleBlocks, 19)
        XCTAssertEqual(FluxDiTConfig.schnell.numSingleBlocks, 38)
        XCTAssertFalse(FluxDiTConfig.schnell.guidanceEmbed,
            "Schnell does not use CFG guidance embed")

        XCTAssertEqual(FluxDiTConfig.dev.numDoubleBlocks, 19)
        XCTAssertTrue(FluxDiTConfig.dev.guidanceEmbed,
            "Dev uses CFG guidance embed")

        // Z-Image Turbo is narrower+shallower than Flux Schnell so the
        // ~2B param budget holds. If someone bumps these accidentally
        // the UI will OOM on M-series machines.
        XCTAssertLessThan(
            FluxDiTConfig.zImageTurbo.numDoubleBlocks,
            FluxDiTConfig.schnell.numDoubleBlocks,
            "Z-Image Turbo should have fewer blocks than Flux Schnell"
        )
        XCTAssertLessThan(
            FluxDiTConfig.zImageTurbo.numSingleBlocks,
            FluxDiTConfig.schnell.numSingleBlocks
        )
    }

    // MARK: - WanDiTConfig presets

    func testWanDiTConfigPresetsSanity() {
        XCTAssertEqual(WanDiTConfig.wan21_1_3B.numLayers, 24)
        XCTAssertEqual(WanDiTConfig.wan21_14B.numLayers, 30)
        XCTAssertLessThan(
            WanDiTConfig.wan21_1_3B.dim,
            WanDiTConfig.wan21_14B.dim,
            "1.3B narrower than 14B"
        )
        // Patch layout defaults for video: temporal 1, spatial 2x2.
        XCTAssertEqual(WanDiTConfig.wan21_14B.patchSizeT, 1)
        XCTAssertEqual(WanDiTConfig.wan21_14B.patchSizeH, 2)
        XCTAssertEqual(WanDiTConfig.wan21_14B.patchSizeW, 2)
    }

    // MARK: - VAE constants

    func testVAEFluxScaleShiftConstants() {
        // Flux VAE rescale factors — changing these breaks every Flux
        // model. Lock them in.
        XCTAssertEqual(VAEDecoder.fluxScaleFactor, 0.3611, accuracy: 1e-4)
        XCTAssertEqual(VAEDecoder.fluxShiftFactor, 0.1159, accuracy: 1e-4)
    }

    func testWanVAEScaleShiftConstants() {
        XCTAssertEqual(WanVAEDecoder.wanScaleFactor, 0.2, accuracy: 1e-4)
        XCTAssertEqual(WanVAEDecoder.wanShiftFactor, 0.0, accuracy: 1e-4)
    }
}
