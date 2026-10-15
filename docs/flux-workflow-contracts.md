# Image workflow contracts

Qwen-Image 2.1 and Turbo expose configuration-only limits and effective completion parameters. Other backends continue returning `nil` metadata until they implement the same contract; the facade does not fabricate values from a request.

```swift
let limits = try await engine.limits(for: bundleURL, name: "qwen-image-2.1-turbo")
try limits.validate(request) // before engine.load; no tensors allocated
for try await event in await engine.generate(request) {
    if case .completed(let url, let seed, let effective) = event {
        // Persist effective with the result. It describes the actual pipeline.
    }
}
```

The optional canonical name supports renamed bundle directories. Without it the query uses the same name routing as the local model store. A missing implementation throws `FluxError.notImplemented`; a registered loader alone is not proof that an operation or limits query is implemented.

Limits distinguish architecture from host admission. Qwen supports multiples of 32, at least two denoising steps, PNG output, and up to ten edit references. Osaurus retains its smaller tested capacity envelope (2048 maximum dimension and four references). Limits with no fixed maximum are not a promise that a request fits available memory. Turbo's fixed grid controls its actual step count.

Invalid fields throw `FluxError.invalidField(field:value:allowed:)`. Qwen requests no longer silently ignore extra images, non-PNG output, masks, partial editing strength, or a negative prompt without CFG. Native batching remains a separate implementation. `ImageEditRequest.strength` is optional: omission means the backend's native edit operation; Qwen also accepts the equivalent value 1.

Completion now has a third associated value, `effectiveParameters`, defaulting to nil for existing backend constructors. Consumers must update exhaustive `.completed` patterns. Qwen reports resolved dimensions, step count, guidance, seed, Euler sampler, scheduler identity, and the bundle sigma grid when present. Revision is absent unless known; it is never inferred from a directory name. Explicit and random seed selection and the sampling arithmetic are unchanged.

Native SeedVR2 and its cooperative producer interruption are implemented separately in #597; see [seedvr2-runtime.md](seedvr2-runtime.md). Existing generation/edit producers still drain on cancellation; this does not claim immediate interruption for every backend. Native batching, selectable samplers, previews, request memory estimates, LoRA, WAN video, and granular latent/conditioning handles remain separate work.

Osaurus #3095 preserves these parameters in panel results, HTTP/base64 and SSE responses, agent image-tool results and Insights. [Live panel/API proof](https://github.com/osaurus-ai/osaurus/pull/3095#issuecomment-6182099288) verified that requested 40 versus effective 8 Turbo steps produced byte-identical output at seed 7, with invalid dimensions rejected before load. Other image families still report absent metadata until independently implemented; no replay guarantee is inferred from a legacy placeholder seed.
