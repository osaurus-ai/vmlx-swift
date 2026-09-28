# Routed activation identity

A closure cannot be identified by its value at one input. Previously `SwitchGLU` compared an activation at 1 against SiLU and approximate GELU, then replaced matching closures with those fast paths. A custom function can agree at 1 and differ everywhere else; constructor execution also unexpectedly evaluated caller code and synchronized the GPU.

`SwitchGLUActivation` now identifies known SiLU and approximate GELU explicitly. An existing caller that supplies a closure retains source compatibility through the closure initializer, which records a custom activation and always preserves it. The default remains known SiLU. Existing calls remain source-compatible. The class is public, not open, and the current source inventory contains no generic SwitchGLU subclasses. Three Gemma constructors explicitly identify approximate GELU, preserving their fast paths without probing. Glue and scored-glue precedence is unchanged.

The exact generic `SwitchGLU` constructor inventory contains 35 production callsites at the audit base. Twenty-nine use the default; six supply an activation. Of those six, three are the Gemma approximate-GELU calls, one is DeepSeek-V3's local clipped helper, one is GLM5Next's independently corrected two-input clamp, and one is DeepSeek-V4's SiLU alongside glue/scoredGlue. TurboQuant-specific types are separate implementations and not included in this count.

## DeepSeek-V3 contract

The vendor revision [`e815299b0bcbac849fa540c768ef21845365c9eb`](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py) uses `ACT2FN[config.hidden_act]` in the MLP shared by routed and shared experts. Its [configuration](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json) specifies SiLU. The local routed-only ±100 output clip does not belong to that contract and was being bypassed by the old probe. Remove that helper instead of inadvertently activating a hidden behavior change. Shared and JANGTQ paths already use plain SiLU.

## Proof plan and limits

The baseline-compatible custom-activation suite covers functions equal to SiLU/GELU at 1 but different at negative/positive inputs, custom-constructor side effects, and glue/scoredGlue precedence. It exercises actual generic SwitchGLU projections both floating and affine 4-bit/g64, decode and sorted prefill. The candidate-only suite checks the known SiLU/GELU identities and outputs. Execution is pending; source alone is not runtime proof.

The specialized `qwen4ExpReduced` entry point retains its existing explicit caller-selected kernel/metadata contract. This change does not infer compatibility of arbitrary two-input glue from `swigluLimit`, broaden fused reducer eligibility, or claim JANGH runtime parity. Model-family, chip and app proof remain separate gates.
