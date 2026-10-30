# Vendored Jinja compatibility changes

The SwiftPM target `VMLXJinja` uses `Sources/Jinja` in this directory. Historical external-fork notes in `Libraries/MLXLMCommon/ChatTemplates/swift-jinja-patches` do not describe the current package source.

## Namespace attribute identity

`namespace(items=[])` must expose its stored `items` list, not dictionary `items()`. Namespace values have their own runtime case and attribute lookup. Mapping methods retain their existing precedence and call behavior. Namespace assignment continues through the nearest environment scope so loop accumulation persists.

This prevents native chat templates that accumulate tool-result blocks from attempting to add a function to a list. It changes the interpreter, not model templates, prompts, generation settings, or tool output.

Regression coverage: `JinjaNamespaceMemberTests` covers method-name collisions, loop accumulation, dictionary method controls, and namespace type/truth semantics. Preserve these tests when updating the vendored interpreter; remove this compatibility change only when equivalent behavior is present upstream.
