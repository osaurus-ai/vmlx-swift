// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// `patch_size` in a preprocessor config is spelled both as a number and as a {height, width}
// object, within the same model family. The object form made the processor configuration
// undecodable, so the model could not load — found by running a real bundle, not by reading types.

import Foundation
import MLXLMCommon
import MLXVLM
import Testing

@Suite("Processor patch_size accepts either spelling")
struct ProcessorPatchSizeShapeTests {

    private func decode(_ json: String) throws -> IntOrSquareSize {
        try JSONDecoder().decode(IntOrSquareSize.self, from: Data(json.utf8))
    }

    @Test("a bare number decodes")
    func scalarDecodes() throws {
        #expect(try decode("16").value == 16)
    }

    /// The shape that used to fail. Magistral-Small-2509 ships exactly this.
    @Test("a square {height, width} object decodes")
    func squareObjectDecodes() throws {
        #expect(try decode(#"{"height": 14, "width": 14}"#).value == 14)
    }

    @Test("a half-specified object takes the dimension it has")
    func partialObjectDecodes() throws {
        #expect(try decode(#"{"height": 14}"#).value == 14)
        #expect(try decode(#"{"width": 12}"#).value == 12)
    }

    /// Deliberately an error rather than a guess: every caller uses this value for BOTH axes, so
    /// picking one would compute a wrong patch grid rather than an approximate one.
    @Test("a non-square pair is refused, not silently halved")
    func nonSquareRefused() {
        #expect(throws: (any Error).self) {
            _ = try decode(#"{"height": 14, "width": 16}"#)
        }
    }

    @Test("an unrelated shape is refused")
    func garbageRefused() {
        #expect(throws: (any Error).self) { _ = try decode(#""fourteen""#) }
    }

    /// A REAL object-form preprocessor config, kept as a fixture.
    ///
    /// Verbatim from Magistral-Small-2509-MLX-8bit, which was the only bundle on the machine that
    /// spelled `patch_size` as an object — 1 of 52. Once that 24 GB model is deleted the
    /// real-bundle walk below has no positive case left, and would keep passing while covering
    /// nothing. 667 bytes is a cheap way to make the coverage outlive the download.
    private static let objectFormPreprocessorConfig = """
        {
          "crop_size": null,
          "data_format": "channels_first",
          "default_to_square": true,
          "device": null,
          "do_center_crop": null,
          "do_convert_rgb": true,
          "do_normalize": true,
          "do_rescale": true,
          "do_resize": true,
          "image_mean": [
            0.48145466,
            0.4578275,
            0.40821073
          ],
          "image_processor_type": "PixtralImageProcessor",
          "image_std": [
            0.26862954,
            0.26130258,
            0.27577711
          ],
          "input_data_format": null,
          "patch_size": {
            "height": 14,
            "width": 14
          },
          "processor_class": "PixtralProcessor",
          "resample": 3,
          "rescale_factor": 0.00392156862745098,
          "return_tensors": null,
          "size": {
            "longest_edge": 1540
          }
        }
        """

    @Test("the real object-form config from a bundle decodes")
    func realObjectFormFixtureDecodes() throws {
        let cfg = try JSONDecoder().decode(
            Mistral3VLMProcessorConfiguration.self,
            from: Data(Self.objectFormPreprocessorConfig.utf8))
        #expect(cfg.imageProcessor.patchSize == 14)
    }

    /// The real preprocessor configs on this machine, which is what caught this.
    @Test("every local mistral3/pixtral preprocessor config decodes")
    func realProcessorConfigsDecode() throws {
        let root = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/MLModels")
        let fm = FileManager.default
        guard let orgs = try? fm.contentsOfDirectory(atPath: root.path) else { return }
        var checked = 0, objectForm = 0
        for org in orgs.sorted() {
            let orgDir = root.appendingPathComponent(org)
            guard let kids = try? fm.contentsOfDirectory(atPath: orgDir.path) else { continue }
            for kid in kids.sorted() {
                let dir = orgDir.appendingPathComponent(kid)
                guard let cfg = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
                    let obj = (try? JSONSerialization.jsonObject(with: cfg)) as? [String: Any],
                    obj["model_type"] as? String == "mistral3",
                    let pre = try? Data(
                        contentsOf: dir.appendingPathComponent("preprocessor_config.json")),
                    let preObj = (try? JSONSerialization.jsonObject(with: pre)) as? [String: Any]
                else { continue }
                if preObj["patch_size"] is [String: Any] { objectForm += 1 }
                _ = try JSONDecoder().decode(
                    Mistral3VLMProcessorConfiguration.self, from: pre)
                checked += 1
            }
        }
        print("preprocessor configs decoded: \(checked)  (object-form patch_size: \(objectForm))")
    }
}
