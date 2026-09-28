# Routed activation identity

A closure cannot be identified by its value at one input. Previously `SwitchGLU` compared an activation at 1 against SiLU and approximate GELU, then replaced matching closures with those fast paths. A custom function can agree at 1 and differ everywhere else; constructor execution also unexpectedly evaluated caller code and synchronized the GPU.

`SwitchGLUActivation` now identifies known SiLU and approximate GELU explicitly. An existing caller that supplies a closure retains source compatibility through the closure initializer, which records a custom activation and always preserves it. The default remains known SiLU. Existing calls remain source-compatible. The class is public, not open, and the current source inventory contains no generic SwitchGLU subclasses. Three Gemma constructors explicitly identify approximate GELU, preserving their fast paths without probing. Glue and scored-glue precedence is unchanged.

The exact generic `SwitchGLU` constructor inventory contains 35 production callsites at the audit base. Twenty-nine use the default; six supply an activation. Of those six, three are the Gemma approximate-GELU calls, one is DeepSeek-V3's local clipped helper, one is GLM5Next's independently corrected two-input clamp, and one is DeepSeek-V4's SiLU alongside glue/scoredGlue. TurboQuant-specific types are separate implementations and not included in this count.

## DeepSeek-V3 contract

The vendor revision [`e815299b0bcbac849fa540c768ef21845365c9eb`](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py) uses `ACT2FN[config.hidden_act]` in the MLP shared by routed and shared experts. Its [configuration](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json) specifies SiLU. The local routed-only ±100 output clip does not belong to that contract and was being bypassed by the old probe. Remove that helper instead of inadvertently activating a hidden behavior change. Shared and JANGTQ paths already use plain SiLU.

## Fused reduction contract

`SwitchGLUActivation.swiGLU(limit:)` declares the two-input function used by GLM: upper-bound the raw gate, bound raw up on both sides, then apply SiLU and multiply. A nil limit leaves both projections unclamped. This descriptor supplies both the eager function and reducer limit. Explicit opaque glue retains precedence and disables reduction; scored glue also disables it. Custom and GELU activations cannot qualify for the SwiGLU reducer through projection geometry alone. Plain SiLU remains eligible when no conflicting glue or independent clamp metadata is supplied.

## Validation and remaining gates

The baseline-compatible custom suite exercises functions that agree with SiLU/GELU at 1 but differ elsewhere, constructor side effects, and glue/scored-glue precedence. It uses actual floating and exactly representable affine 4-bit identity projections, decode and sorted prefill. Known-identity tests preserve SiLU/GELU fast paths; reducer tests cover qualified geometry with unknown math rejected and known typed math retained.

An integration build at `df99ffece51054251e30cee17ccdfc5c6ee6a585` passed all four custom, three known, and three GLM tests with TF32 explicitly disabled for diagnosis. Its default-precision failures were confined to the fixture's assumption that a floating identity down projection preserves every input bit under sorted TF32 arithmetic. The corrected oracle computes expected activation independently on the CPU, then passes those rows through the real down projection and matching sort/unsort. Tolerances remain unchanged. Decode and TF32-disabled diagnostic runs also retain scalar output checks. Runtime precision defaults are unchanged.

Default-precision execution of the revised oracle and the new reducer-contract tests is pending. The preceding integration result is not an independent build of this branch. Model-family, chip, app, and performance proof remain separate; this change does not establish JANGH runtime support.
