# GLM native history and text cache boundaries

GLM's message generator previously reconstructed only role and media content. That discarded reasoning, tool calls, result IDs and emitted argument order before the native template could interpret them. A dependent read/write conversation could therefore render as ordinary assistant text followed by unassociated tool results.

The generator now starts from `defaultMessageDict(for:)` and replaces only GLM's media content items. The existing text/image/video item order is preserved. The active bundle template still decides whether reasoning is retained or cleared; no template, parser, sampler, projection or kernel arithmetic changes are included.

For text-only input the processor publishes token IDs, tool schemas and the existing canonical/stable cache-boundary helper's exact-prefix proofs. Unsupported tokenizers fail closed. Image/video/audio companion boundaries are not inferred from the text path, and this change does not alter quota selection or persisted cache schema.

## Regression coverage

`Glm5NextMessageHistoryTests` checks reasoning and two dependent calls, call/result IDs, typed arguments, emitted argument ordering after Codable persistence, and metadata preservation when media markers are present.

`ActualGlm5NextHistoryTokenizerTests` requires local GLM native sidecars. It checks native reasoning policy, read/result/write order, persisted argument order, result-ID reordering, three growing-history prefixes, per-tool continuation prefixes, and image/video marker order. These tests do not load model weights or decode media payloads.

The retained native regression ran all nine methods, with zero skips: old mapping source `3db9e5627c355f2abd0a6c784f783cf37e30a4f2` produced eight failures plus the expected media-marker control pass; corrected mapping source `9bba092a70b62f5be2042748d6483502aa38e127` produced nine passes. These are integration-source fixture results, not a build or GUI result for this later main-based delivery branch.

The corrected growing prompts had token counts 7621/9281/10106; the dependent-tool fixture had counts 200/245/293. The longer fixture elapsed time included a 90-second shared test-semaphore wait and is not tokenizer or application performance evidence.

Retained evidence identities:

| Evidence | SHA256 |
| --- | --- |
| Old-mapping native log | `d70ae938df4af47b05f791f6137bf4241d2f64560520316f52fc8a1b377bd1f7` |
| Corrected native log | `08cd140187828f3166ffbf899c2de839060acf054d8fb5696c1b49cce4ac6a0c` |
| Corrected native receipt | `32d43bc28945a9cc137a84ddf5579b275acd5fbfc2398bf957fedebd9a8004d6` |

The logs are retained evidence, not publicly downloadable artifacts. This branch rebases the same three source/test files onto engine main `44862aab733a4ca68d9bc66455e54ff7c2c61593`; its exact build, focused rerun and consuming-app proof remain pending.

## Public synthetic reproducer

No retained conversation is needed. Supply the real local GLM tokenizer sidecars and a synthetic three-turn history matching the test's JSON schema:

```sh
glm_fixture_dir=$(mktemp -d)
cat >"$glm_fixture_dir/history.json" <<'JSON'
{
  "turns": [
    {
      "user": "For this example, the note identifier is ALPHA_07. Remember that identifier.",
      "visible": "The note identifier is ALPHA_07.",
      "reasoning": "Retain the identifier supplied in this example."
    },
    {
      "user": "What identifier did I provide?",
      "visible": "You provided ALPHA_07.",
      "reasoning": "Use the identifier from the previous message."
    },
    {
      "user": "Repeat the same identifier once more.",
      "visible": "ALPHA_07.",
      "reasoning": "The identifier remains unchanged in this example."
    }
  ]
}
JSON
export VMLX_GLM5_NATIVE_TOKENIZER_DIR="/path/to/GLM-5.3-Flash-JANGH2"
export VMLX_GLM5_NATIVE_CONVERSATION_RECEIPT="$glm_fixture_dir/history.json"
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test --jobs 1 \
  --filter 'Glm5NextMessageHistoryTests|ActualGlm5NextHistoryTokenizerTests'
```

The tokenizer directory must contain `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and `processor_config.json`. Use a native macOS test host with the MLX Metal library packaged correctly. Require all nine named methods to execute and pass, with zero skips; a successful process with no selected tests is not proof. This public synthetic input is a new reproduction input and has not been run for this source port.

The per-tool fixture renders histories after the first user message and after each dependent tool result. It verifies complete message/token-prefix construction at each boundary. It does not by itself prove an emitted live tool, SSD publication/restore, KDA recurrent-companion reconstruction, MTP, media payload processing or sustained decode speed. Those remain separate consuming-runtime gates.
