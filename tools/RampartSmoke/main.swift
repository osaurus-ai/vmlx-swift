// Rampart PII NER correctness smoke.
//
// Loads the Rampart MLX model, runs detection on a few strings (or one passed
// via RAMPART_TEXT), and prints detected PII spans + a redacted line.
//
// Usage:
//   RAMPART_MODEL=/path/to/rampart-mlx swift run RampartSmoke
//   RAMPART_MODEL=/path/to/rampart-mlx RAMPART_TEXT="email me at a@b.com" swift run RampartSmoke
//
// Requires the MLX metallib next to the executable (see VMLX_README).

import Foundation
import RampartPII

@main
struct RampartSmoke {
    static func main() throws {
        setvbuf(stdout, nil, _IONBF, 0)
        let env = ProcessInfo.processInfo.environment
        let modelPath = env["RAMPART_MODEL"] ?? "/tmp/rampart-mlx"

        let dir = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("model.safetensors").path)
        else {
            fputs("model.safetensors not found in \(dir.path)\n", stderr)
            exit(1)
        }

        print("[rampart] loading \(dir.lastPathComponent) ...")
        let start = CFAbsoluteTimeGetCurrent()
        let pii = try RampartPII(directory: dir)
        print(String(format: "[rampart] loaded in %.2fs", CFAbsoluteTimeGetCurrent() - start))

        let texts: [String]
        if let one = env["RAMPART_TEXT"] {
            texts = [one]
        } else {
            texts = [
                "My name is John Smith and my email is john.smith@example.com",
                "Call me at (555) 123-4567 or visit 42 Main Street, Springfield, IL 62704",
                "The meeting is at 3pm in the main conference room.",
            ]
        }

        for text in texts {
            print("\n>>> \(text)")
            let spans = pii.detect(text)
            if spans.isEmpty {
                print("    (no PII)")
            }
            for s in spans {
                print(String(format: "    %@  %@  [%d:%d]  %.2f",
                    s.type.padding(toLength: 16, withPad: " ", startingAt: 0),
                    s.text, s.range.lowerBound, s.range.upperBound, s.score))
            }
            print("    redacted: \(pii.redact(text))")
        }
    }
}
