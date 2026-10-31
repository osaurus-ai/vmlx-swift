# Vendored Jinja compatibility changes

The SwiftPM target `VMLXJinja` uses `Sources/Jinja` in this directory. Historical external-fork notes in `Libraries/MLXLMCommon/ChatTemplates/swift-jinja-patches` do not describe the current package source.

## Namespace attribute identity

`namespace(items=[])` must expose its stored `items` list, not dictionary `items()`. Namespace values have their own runtime case and attribute lookup. Mapping methods retain their existing precedence and call behavior. Namespace assignment continues through the nearest environment scope so loop accumulation persists.

This prevents native chat templates that accumulate tool-result blocks from attempting to add a function to a list. It changes the interpreter, not model templates, prompts, generation settings, or tool output.

Regression coverage: `JinjaNamespaceMemberTests` covers method-name collisions, loop accumulation, dictionary method controls, and namespace type/truth semantics. Preserve these tests when updating the vendored interpreter; remove this compatibility change only when equivalent behavior is present upstream.

## Hugging Face JSON and schema scalar semantics

`tojson` follows the transformers chat-template filter's Python JSON contract:
Unicode is preserved by default, separators include spaces, explicit indentation
and separators are honored, and non-JSON values throw instead of silently becoming
null. Jinja object order is retained unless sorting is requested; unordered host
dictionaries retain their existing deterministic ordering. Foundation booleans and
boxed doubles preserve their types. `upper` stringifies schema values as Jinja does
instead of requiring callers to rewrite valid JSON Schema.

`Tests/JinjaReferenceTests` pins the reference and controls, including error cases.
Run `bash .github/scripts/run-jinja-reference-tests.sh` for these CPU-only tests.
Keep this contract and its tests when updating the vendored interpreter. These
tests establish rendering parity, not model-output quality or tool-name fidelity.

## Explicit whitespace trimming

Apply `{{-`, `{%-`, `-}}` and `-%}` at real lexer delimiters, after text has
been tokenized. Source-wide replacement previously inserted a synthetic space
after a literal opening brace and rewrote delimiter-looking text inside quoted
strings. Keeping the literal text and delimiter as separate tokens avoids both
errors without modifying model templates. `JinjaWhitespaceReferenceTests` guards
literal braces, quoted delimiter text, ordinary whitespace and subtraction.
