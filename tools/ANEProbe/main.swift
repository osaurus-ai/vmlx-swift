import Foundation
import MLXLMCommon

#if canImport(CoreML)
import CoreML
#endif

@main
enum ANEProbe {
    static func main() {
        print("VMLX_ANE_PROBE_VERSION=1")
        let mode = AccelerationRuntime.requestedMode()
        print("VMLX_ACCELERATOR_REQUESTED=\(mode.flagValue)")
        do {
            let decision = try AccelerationRuntime.resolveTextDecode(mode)
            switch decision {
            case .metal(let reason):
                print("VMLX_TEXT_DECODE_ACCELERATOR=metal")
                print("VMLX_TEXT_DECODE_REASON=\(reason)")
            case .coreMLANE(let manifestID):
                print("VMLX_TEXT_DECODE_ACCELERATOR=ane-coreml")
                print("VMLX_TEXT_DECODE_MANIFEST=\(manifestID)")
            }
        } catch {
            print("VMLX_TEXT_DECODE_ACCELERATOR=unavailable")
            print("VMLX_TEXT_DECODE_ERROR=\(error.localizedDescription)")
        }

        #if canImport(CoreML)
        print("COREML_AVAILABLE=YES")
        print("MLX_DIRECT_ANE_DEVICE=NO")

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        print("COREML_COMPUTE_UNITS=all")

        if #available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, *) {
            let devices = MLComputeDevice.allComputeDevices
            print("COREML_COMPUTE_DEVICE_COUNT=\(devices.count)")
            for (index, device) in devices.enumerated() {
                print("COREML_COMPUTE_DEVICE_\(index)=\(String(describing: device))")
            }
            let hasNeuralEngine = devices.contains {
                String(describing: $0).contains("MLNeuralEngineComputeDevice")
            }
            print("COREML_NEURAL_ENGINE_VISIBLE=\(hasNeuralEngine ? "YES" : "NO")")
        } else {
            print("COREML_COMPUTE_DEVICE_COUNT=unavailable")
            print("COREML_NEURAL_ENGINE_VISIBLE=UNKNOWN")
        }

        print("RECOMMENDATION=Do not enable an ANE runtime flag until a Core ML island has parity and benchmark logs.")
        #else
        print("COREML_AVAILABLE=NO")
        print("MLX_DIRECT_ANE_DEVICE=NO")
        print("COREML_NEURAL_ENGINE_VISIBLE=NO")
        print("RECOMMENDATION=Core ML is unavailable on this platform; keep MLX/Metal runtime.")
        #endif
    }
}
