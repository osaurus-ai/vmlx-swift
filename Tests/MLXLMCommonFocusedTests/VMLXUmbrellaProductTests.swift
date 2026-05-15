// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Testing
import VMLX

@Suite("VMLX umbrella product")
struct VMLXUmbrellaProductTests {
    @Test("umbrella re-exports Osaurus runtime modules")
    func reexportsRuntimeModules() {
        let _: MLXArray.Type = MLXArray.self
        let _: ModelContext.Type = ModelContext.self
        let _: UserInput.Type = UserInput.self
        let _: GenerateParameters.Type = GenerateParameters.self
        let _: LLMRegistry.Type = LLMRegistry.self
        let _: MediaProcessing.Type = MediaProcessing.self
        let _: AutoTokenizer.Type = AutoTokenizer.self
        let _: Template.Type = Template.self
        let _: VMLXServerRuntimeSettings.Type = VMLXServerRuntimeSettings.self
    }
}
