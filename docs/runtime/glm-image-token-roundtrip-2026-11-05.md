# GLM required image-token resolution

The GLM processor accepted a non-nil vocabulary lookup as its required image token. Tokenizers may return an unknown-token alias, while inconsistent tokenizers can split the placeholder literal or decode its ID to different text. Such an ID must not become an image/cache placeholder.

Use the existing media-token resolver for the required `<|image|>` literal. It encodes without added special tokens, requires exactly one ID and checks the exact decoded literal. Invalid required placeholders keep the existing constructor error; valid explicit-ID construction and optional image/video delimiter lookup are unchanged. No sampler, prompt, model weights, AgentLoop or app-tool behavior changes are included.

The normal Xcode baseline reproduced three negative variants: unknown alias, split literal and mismatched decode were all incorrectly accepted. After the correction, all three constructor test functions pass, including four invalid-token arguments and the genuine-token registry control. Five separate checks using the actual local GLM tokenizer pass for native reasoning history, dependent tool argument/result order, canonical growing prefixes and media marker mapping. The first native-tokenizer attempt was skipped because Xcode did not forward a plain environment variable; it is not counted. The successful follow-up uses explicit test-runner forwarding. Canonical sidecar hashes and mtimes are unchanged. No model weights were loaded in these component tests.

Source-bound dev-app proof and normal PR checks are pending. These results do not qualify GLM inference quality, speed, RAM, real media payloads or all optional placeholder resolution. No release or tag is included.
