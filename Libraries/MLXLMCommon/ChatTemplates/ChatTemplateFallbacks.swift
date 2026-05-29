// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Hard-coded fallback chat templates. Used by the huggingface tokenizer
// bridge when the model's native `chat_template.jinja` triggers a
// swift-jinja parser/runtime bug it can't evaluate. The bridge tries
// each candidate in order and uses the first one that renders;
// downstream consumers see no difference.
//
// Known upstream limitations these templates work around
// (johnmai-dev/Jinja 1.3.x):
//
//   - Gemma-4 templates (26B-A4B-it-*, E2B/E4B, 31B-JANG_4M):
//     `JinjaError.syntax("Unexpected token: multiplicativeBinaryOperator")`
//     at parse. Individual constructs parse fine but the full template
//     assembly trips the parser.
//
//   - Nemotron-Cascade-2 templates: `JinjaError.runtime("Unknown
//     operation type: not in")`. Root cause is in swift-jinja's for-
//     loop runtime — when iterating a dict with `for k in d`, the loop
//     var is bound to an `ArrayValue([key, value])` rather than the
//     scalar key, and the `not in` containment check hits the
//     ArrayValue × ArrayValue branch which only handles `+`.
//
// These fallbacks are intentionally minimal — they keep the prompt
// contract (role markers, tool declaration, generation-prompt suffix)
// but drop the complex formatting (BNF-style parameter blocks, etc.)
// that's either upstream-bug territory or purely cosmetic.

import Foundation

public enum ChatTemplateFallbacks {

    /// Gemma-4 text-only + image / video / audio, no tools. Preserves
    /// `<|turn>role` / `<turn|>` delimiters that the Gemma-4 model
    /// family was trained on.
    public static let gemma4Minimal: String = #"""
{{- bos_token -}}
{%- macro render_content(content) -%}
    {%- if content is string -%}
        {{- content | trim -}}
    {%- elif content is sequence -%}
        {%- for item in content -%}
            {%- if item['type'] == 'text' -%}
                {{- item['text'] | trim -}}
            {%- elif item['type'] == 'image' -%}
                {{- '\n\n<|image|>\n\n' -}}
            {%- elif item['type'] == 'video' -%}
                {{- '\n\n<|video|>\n\n' -}}
            {%- elif item['type'] == 'audio' -%}
                {{- '\n\n<|audio|>\n\n' -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}
{%- if messages[0]['role'] == 'system' -%}
    {{- '<|turn>system\n' -}}
    {{- render_content(messages[0]['content']) -}}
    {{- '<turn|>\n' -}}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
    {{- '<|turn>' + role + '\n' -}}
    {{- render_content(message['content']) -}}
    {{- '<turn|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|turn>model\n' -}}
{%- endif -%}
"""#

    /// Gemma-4 with a `<|tool>...<tool|>` declaration block for each
    /// tool and `<|tool_call>call:name{args}<tool_call|>` assistant
    /// output. Tool replies render as
    /// `<|tool_response>response:name{content}<tool_response|>`.
    public static let gemma4WithTools: String = #"""
{{- bos_token -}}
{%- macro render_content(content) -%}
    {%- if content is string -%}
        {{- content | trim -}}
    {%- elif content is sequence -%}
        {%- for item in content -%}
            {%- if item['type'] == 'text' -%}
                {{- item['text'] | trim -}}
            {%- elif item['type'] == 'image' -%}
                {{- '\n\n<|image|>\n\n' -}}
            {%- elif item['type'] == 'video' -%}
                {{- '\n\n<|video|>\n\n' -}}
            {%- elif item['type'] == 'audio' -%}
                {{- '\n\n<|audio|>\n\n' -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}
{%- macro render_tool_args(arguments) -%}
    {%- if arguments is mapping -%}
        {%- set first = true -%}
        {%- for key, value in arguments | dictsort -%}
            {%- if not first %},{% endif -%}
            {%- set first = false -%}
            {{- key -}}:{%- if value is string -%}<|"|>{{ value }}<|"|>
                {%- elif value is boolean -%}{{ 'true' if value else 'false' }}
                {%- else -%}{{ value }}
            {%- endif -%}
        {%- endfor -%}
    {%- elif arguments is string -%}
        {{- arguments -}}
    {%- endif -%}
{%- endmacro -%}
{%- set required_tool_choice = false -%}
{%- set required_tool_name = '' -%}
{%- if tool_choice is defined and tool_choice == 'required' -%}
    {%- set required_tool_choice = true -%}
{%- elif additionalContext is defined and additionalContext['tool_choice'] == 'required' -%}
    {%- set required_tool_choice = true -%}
{%- endif -%}
{%- if tool_choice_name is defined -%}
    {%- set required_tool_name = tool_choice_name -%}
{%- elif additionalContext is defined and additionalContext['tool_choice_name'] is defined -%}
    {%- set required_tool_name = additionalContext['tool_choice_name'] -%}
{%- endif -%}
{%- if required_tool_choice and not required_tool_name and tools is iterable and tools | length == 1 -%}
    {%- set only_required_tool = tools[0]['function'] if tools[0]['function'] is defined else tools[0] -%}
    {%- if only_required_tool['name'] is defined -%}
        {%- set required_tool_name = only_required_tool['name'] -%}
    {%- endif -%}
{%- endif -%}
{%- macro render_required_tool_choice_instruction(latest_user_content='') -%}
    {%- if required_tool_choice -%}
        {{- '\nThe current assistant response MUST be a function call. Reply only with one native Gemma function call and no prose before the tool result:\n<|tool_call>call:FUNCTION_NAME{ARGUMENT_NAME:<|"|>ARGUMENT_VALUE<|"|>}<tool_call|>' -}}
        {%- if required_tool_name -%}
            {{- '\nUse the `' ~ required_tool_name ~ '` function.' -}}
            {%- for tool in tools -%}
                {%- set selected_tool = tool['function'] if tool['function'] is defined else tool -%}
                {%- if selected_tool['name'] == required_tool_name and selected_tool['parameters'] is defined and selected_tool['parameters']['required'] is defined -%}
                    {{- '\nRequired parameters for `' ~ required_tool_name ~ '`: ' ~ (selected_tool['parameters']['required'] | join(', ')) ~ '.' -}}
                    {%- for param_name in selected_tool['parameters']['required'] -%}
                        {%- set exact = namespace(value='') -%}
                        {%- set exact_markers = [
                            'on this exact ' ~ param_name ~ ':',
                            'this exact ' ~ param_name ~ ':',
                            'exact ' ~ param_name ~ ':',
                            'on exactly this ' ~ param_name ~ ':',
                            'exactly this ' ~ param_name ~ ':',
                            'on this exact text:',
                            'this exact text:',
                            'exact text:',
                            'on exactly this text:',
                            'exactly this text:',
                            'now use ' ~ required_tool_name ~ ' on this exact text:',
                            'use ' ~ required_tool_name ~ ' on this exact text:'
                        ] -%}
                        {%- for marker in exact_markers -%}
                            {%- if not exact.value and latest_user_content is string and marker in latest_user_content -%}
                                {%- set exact.value = latest_user_content.split(marker)[1] | trim -%}
                            {%- endif -%}
                        {%- endfor -%}
                        {%- if exact.value -%}
                            {{- '\nRequired call shape for the current request:\n<|tool_call>call:' ~ required_tool_name ~ '{' ~ param_name ~ ':<|"|>' ~ exact.value ~ '<|"|>}<tool_call|>' -}}
                        {%- endif -%}
                    {%- endfor -%}
                {%- endif -%}
            {%- endfor -%}
        {%- endif -%}
    {%- endif -%}
{%- endmacro -%}
{%- if (tools or (messages[0]['role'] in ['system', 'developer'])) -%}
    {{- '<|turn>system\n' -}}
    {%- if messages[0]['role'] in ['system', 'developer'] -%}
        {{- render_content(messages[0]['content']) -}}
        {%- set loop_messages = messages[1:] -%}
    {%- else -%}
        {%- set loop_messages = messages -%}
    {%- endif -%}
    {%- if tools -%}
        {%- for tool in tools -%}
            {%- set fn = tool['function'] if tool['function'] is defined else tool -%}
            {{- '<|tool>declaration:' + fn['name'] + '{' -}}
            {%- if fn['description'] is defined and fn['description'] -%}
                {{- 'description:<|"|>' + (fn['description'] | trim) + '<|"|>' -}}
            {%- endif -%}
            {%- if fn['parameters'] is defined and fn['parameters']['properties'] is defined -%}
                {%- if fn['description'] is defined and fn['description'] -%}
                    {{- ',' -}}
                {%- endif -%}
                {{- 'parameters:{' -}}
                {%- set first_param = true -%}
                {%- for param_name, param in fn['parameters']['properties'] | dictsort -%}
                    {%- if not first_param -%}
                        {{- ',' -}}
                    {%- endif -%}
                    {%- set first_param = false -%}
                    {{- param_name + ':{' -}}
                    {%- if param['type'] is defined -%}
                        {{- 'type:<|"|>' -}}
                        {%- if param['type'] is string -%}
                            {{- param['type'] -}}
                        {%- else -%}
                            {{- param['type'] | tojson -}}
                        {%- endif -%}
                        {{- '<|"|>' -}}
                    {%- endif -%}
                    {%- if param['description'] is defined -%}
                        {%- if param['type'] is defined -%}
                            {{- ',' -}}
                        {%- endif -%}
                        {{- 'description:<|"|>' + (param['description'] | trim) + '<|"|>' -}}
                    {%- endif -%}
                    {{- '}' -}}
                {%- endfor -%}
                {{- '}' -}}
                {%- if fn['parameters']['required'] is defined -%}
                    {{- ',required:' + (fn['parameters']['required'] | tojson) -}}
                {%- endif -%}
            {%- endif -%}
            {{- '}<tool|>' -}}
        {%- endfor -%}
    {%- endif -%}
    {{- '<turn|>\n' -}}
{%- else -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
    {%- if message['role'] == 'tool' -%}
        {{- '<|tool_response>response:' + (message.get('name') or 'unknown') + '{' -}}
        {{- render_content(message['content']) -}}
        {{- '}<tool_response|>\n' -}}
    {%- else -%}
        {{- '<|turn>' + role + '\n' -}}
        {%- if message['content'] -%}
            {{- render_content(message['content']) -}}
        {%- endif -%}
        {%- if message['role'] == 'user' and required_tool_choice and loop.last -%}
            {%- set latest_required_user_content = message['content'] if message['content'] is string else '' -%}
            {{- render_required_tool_choice_instruction(latest_required_user_content) -}}
        {%- endif -%}
        {%- if message['tool_calls'] -%}
            {%- for tool_call in message['tool_calls'] -%}
                {%- set fn = tool_call['function'] -%}
                {{- '<|tool_call>call:' + fn['name'] + '{' -}}
                {{- render_tool_args(fn['arguments']) -}}
                {{- '}<tool_call|>' -}}
            {%- endfor -%}
        {%- endif -%}
        {{- '<turn|>\n' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|turn>model\n' -}}
{%- endif -%}
"""#

    /// Nemotron-Cascade-2 minimal. Avoids the native template's
    /// `for k in dict if k not in handled_keys` construct that trips
    /// swift-jinja's runtime, but preserves the trained prompt contract:
    /// ChatML turns, `# Tools`, `<tools>/<function>` declarations,
    /// `<tool_call>` XML assistant calls, `<tool_response>` tool replies,
    /// and the `<think></think>` generation prefix when thinking is off.
    public static let nemotronMinimal: String = #"""
{%- set loop_messages = messages -%}
{%- set enable_thinking = enable_thinking if enable_thinking is defined else false -%}
{%- set required_tool_choice = false -%}
{%- if tool_choice is defined and tool_choice == 'required' -%}
    {%- set required_tool_choice = true -%}
{%- elif additionalContext is defined and additionalContext['tool_choice'] == 'required' -%}
    {%- set required_tool_choice = true -%}
{%- endif -%}
{%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set system_message = '' -%}
{%- endif -%}
<|im_start|>system
{{ system_message }}
{%- if tools %}

# Tools

You have access to the following functions:

<tools>
{%- for tool in tools %}
  {%- set fn = tool['function'] if tool['function'] is defined else tool -%}
  <function>
    <name>{{ fn['name'] }}</name>
    {%- if fn['description'] is defined %}
    <description>{{ fn['description'] | trim }}</description>
    {%- endif %}
    {%- if fn['parameters'] is defined and fn['parameters']['properties'] is defined %}
    <parameters>
      {%- for param_name, param in fn['parameters']['properties'] | dictsort %}
      <parameter>
        <name>{{ param_name }}</name>
        {%- if param['type'] is defined %}<type>{{ param['type'] }}</type>{%- endif %}
        {%- if param['description'] is defined %}<description>{{ param['description'] | trim }}</description>{%- endif %}
      </parameter>
      {%- endfor %}
      {%- if fn['parameters']['required'] is defined %}
      <required>{{ fn['parameters']['required'] }}</required>
      {%- endif %}
    </parameters>
    {%- endif %}
  </function>
{%- endfor %}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call>
{%- if required_tool_choice %}

<IMPORTANT>
The current assistant response MUST be a tool call. Reply only with a `<tool_call>` block for one available tool and no prose before the tool result.
</IMPORTANT>
{%- endif %}
{%- endif %}
<|im_end|>
{% for message in loop_messages -%}
{%- if message['role'] == 'user' -%}
<|im_start|>user
{{ message['content'] }}
{%- if required_tool_choice and loop.last %}

<IMPORTANT>
The current assistant response MUST be a tool call. This applies to the latest user request. Reply only with a `<tool_call>` block for one available tool and no prose before the tool result.
</IMPORTANT>
{%- endif %}
<|im_end|>
{%- elif message['role'] == 'assistant' -%}
<|im_start|>assistant
{%- if message['content'] -%}
{{ message['content'] }}
{%- endif %}
{%- if message['tool_calls'] is defined and message['tool_calls'] %}
{%- for tc in message['tool_calls'] %}
<tool_call>
<function={{ tc['function']['name'] }}>
{%- if tc['function']['arguments'] is mapping %}
{%- for k, v in tc['function']['arguments'] | dictsort %}
<parameter={{ k }}>
{{ v }}
</parameter>
{%- endfor %}
{%- elif tc['function']['arguments'] is string -%}
{{ tc['function']['arguments'] }}
{%- endif %}
</function>
</tool_call>
{%- endfor %}
{%- endif %}
<|im_end|>
{%- elif message['role'] == 'tool' -%}
<|im_start|>user
<tool_response>
{{ message['content'] }}
</tool_response>
<|im_end|>
{%- endif -%}

{% endfor -%}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- if enable_thinking %}
<think>
{%- else %}
<think></think>
{%- endif %}
{%- endif %}
"""#

    /// DeepSeek-V4 minimal template. DSV4-Flash bundles ship NO
    /// `chat_template` field in `tokenizer_config.json` — the stock
    /// distribution carries an external `encoding/encoding_dsv4.py`
    /// instead. This jinja renders the same wire format the Python
    /// encoder produces (BOS / `<｜User｜>` / `<｜Assistant｜>` /
    /// closed `</think>` chat-mode tail / open `<think>` thinking-
    /// mode tail / DSML tool calls / `enable_thinking=true` +
    /// `reasoning_effort=max` preface).
    /// Selected via the DSV4 BOS sniff in the tokenizer bridge.
    public static let dsv4Minimal: String = #"""
{%- set bos = '<｜begin▁of▁sentence｜>' -%}
{%- set eos = '<｜end▁of▁sentence｜>' -%}
{%- set user_token = '<｜User｜>' -%}
{%- set asst_token = '<｜Assistant｜>' -%}
{%- set think_open = '<think>' -%}
{%- set think_close = '</think>' -%}
{%- set action_token = '<｜action｜>' -%}
{%- set dsml = '｜DSML｜' -%}
{%- set ns = namespace(last_user_index=-1) -%}
{%- for message in messages -%}
{%- if message['role'] == 'user' or message['role'] == 'developer' -%}
{%- set ns.last_user_index = loop.index0 -%}
{%- endif -%}
{%- endfor -%}
{%- macro render_tools(tools) -%}
{{- '\n\n## Tools\n\n' -}}
{{- 'You have access to a set of tools to help answer the user\'s question. You can invoke tools by writing a "<' + dsml + 'tool_calls>" block like the following:\n\n' -}}
{{- '<' + dsml + 'tool_calls>\n' -}}
{{- '<' + dsml + 'invoke name="$TOOL_NAME">\n' -}}
{{- '<' + dsml + 'parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</' + dsml + 'parameter>\n' -}}
{{- '...\n' -}}
{{- '</' + dsml + 'invoke>\n' -}}
{{- '<' + dsml + 'invoke name="$TOOL_NAME2">\n' -}}
{{- '...\n' -}}
{{- '</' + dsml + 'invoke>\n' -}}
{{- '</' + dsml + 'tool_calls>\n\n' -}}
{{- 'For tools with no parameters, emit an empty invoke with no parameter lines:\n\n' -}}
{{- '<' + dsml + 'tool_calls>\n' -}}
{{- '<' + dsml + 'invoke name="$TOOL_NAME_WITHOUT_PARAMETERS">\n' -}}
{{- '</' + dsml + 'invoke>\n' -}}
{{- '</' + dsml + 'tool_calls>\n\n' -}}
{{- 'Do not emit JSON objects for tool calls; tool calls must use DSML invoke blocks.\n\n' -}}
{{- 'String parameters should be specified as is and set `string="true"`. For multiline strings, put real newline characters inside the parameter body; do not write backslash-n escape sequences. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.\n\n' -}}
{{- 'If thinking_mode is enabled (triggered by ' + think_open + '), you MUST output your complete reasoning inside ' + think_open + '...' + think_close + ' BEFORE any tool calls or final response.\n\n' -}}
{{- 'Otherwise, output directly after ' + think_close + ' with tool calls or final response.\n\n' -}}
{{- '### Available Tool Schemas\n\n' -}}
{%- for tool in tools -%}
{%- if tool['function'] is defined -%}
{{- tool['function'] | tojson -}}{{- '\n' -}}
{%- else -%}
{{- tool | tojson -}}{{- '\n' -}}
{%- endif -%}
{%- endfor -%}
{{- '\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.' -}}
{%- if tool_choice is defined and tool_choice == 'required' -%}
{{- '\n\nThe current assistant response MUST be a tool call. Start with a "<' + dsml + 'tool_calls>" block and do not answer in prose before the tool result.' -}}
{%- endif -%}
{%- endmacro -%}
{%- macro render_dsml_invoke(tool_call) -%}
{%- set fn = tool_call['function'] if tool_call['function'] is defined else tool_call -%}
{%- set args = fn['arguments'] if fn['arguments'] is defined else {} -%}
{{- '<' + dsml + 'invoke name="' + fn['name'] + '">\n' -}}
{%- if args is mapping -%}
{%- for k, v in args | dictsort -%}
{%- if v is string -%}
{{- '<' + dsml + 'parameter name="' + k + '" string="true">' + v + '</' + dsml + 'parameter>\n' -}}
{%- else -%}
{{- '<' + dsml + 'parameter name="' + k + '" string="false">' + (v | tojson) + '</' + dsml + 'parameter>\n' -}}
{%- endif -%}
{%- endfor -%}
{%- elif args is string -%}
{{- '<' + dsml + 'parameter name="arguments" string="true">' + args + '</' + dsml + 'parameter>\n' -}}
{%- endif -%}
{{- '</' + dsml + 'invoke>\n' -}}
{%- endmacro -%}
{%- macro render_dsml_tool_calls(tool_calls) -%}
{{- '\n\n<' + dsml + 'tool_calls>\n' -}}
{%- for tool_call in tool_calls -%}
{{- render_dsml_invoke(tool_call) -}}
{%- endfor -%}
{{- '</' + dsml + 'tool_calls>' -}}
{%- endmacro -%}
{{- bos -}}
{%- if enable_thinking and reasoning_effort == 'max' -%}
{{- 'Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n' -}}
{%- endif -%}
{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
{{- message['content'] -}}
{%- if tools -%}
{{- render_tools(tools) -}}
{%- endif -%}
{%- set next_role = messages[loop.index0 + 1]['role'] if loop.index0 + 1 < messages|length else none -%}
{%- if next_role -%}
{{- '\n' -}}
{%- endif -%}
{%- elif message['role'] == 'user' or message['role'] == 'developer' -%}
{%- set prev_role = messages[loop.index0 - 1]['role'] if loop.index0 > 0 else none -%}
{%- if prev_role == 'tool' -%}
{{- '\n\n' -}}{{- message['content'] -}}
{%- else -%}
{{- user_token -}}{{- message['content'] -}}
{%- endif -%}
{%- if tools and not (messages|length > 0 and messages[0]['role'] == 'system') and loop.index0 == ns.last_user_index -%}
{{- render_tools(tools) -}}
{%- endif -%}
{%- set next_role = messages[loop.index0 + 1]['role'] if loop.index0 + 1 < messages|length else none -%}
{%- if next_role == 'assistant' or loop.last and add_generation_prompt -%}
{{- asst_token -}}
{%- if tool_choice is defined and tool_choice == 'required' and loop.index0 >= ns.last_user_index -%}
{{- think_open -}}{{- action_token -}}
{%- elif enable_thinking and loop.index0 >= ns.last_user_index -%}
{{- think_open -}}
{%- else -%}
{{- think_close -}}
{%- endif -%}
{%- endif -%}
{%- elif message['role'] == 'assistant' -%}
{%- if enable_thinking and message.get('reasoning_content') and loop.index0 > ns.last_user_index -%}
{{- message['reasoning_content'] -}}{{- think_close -}}
{%- endif -%}
{{- message['content'] or '' -}}
{%- if message['tool_calls'] is defined and message['tool_calls'] -%}
{{- render_dsml_tool_calls(message['tool_calls']) -}}
{%- endif -%}
{{- eos -}}
{%- elif message['role'] == 'tool' -%}
{%- set prev_role = messages[loop.index0 - 1]['role'] if loop.index0 > 0 else none -%}
{%- if prev_role == 'tool' -%}
{{- '\n\n' -}}
{%- else -%}
{{- user_token -}}
{%- endif -%}
{{- '<tool_result>' -}}{{- message['content'] -}}{{- '</tool_result>' -}}
{%- set next_role = messages[loop.index0 + 1]['role'] if loop.index0 + 1 < messages|length else none -%}
{%- if next_role == 'assistant' or loop.last and add_generation_prompt -%}
{{- asst_token -}}
{%- if tool_choice is defined and tool_choice == 'required' and loop.index0 >= ns.last_user_index -%}
{{- think_open -}}{{- action_token -}}
{%- elif enable_thinking and loop.index0 >= ns.last_user_index -%}
{{- think_open -}}
{%- else -%}
{{- think_close -}}
{%- endif -%}
{%- endif -%}
{%- endif -%}
{%- endfor -%}
"""#

    /// Laguna / Poolside minimal chat template.
    ///
    /// Real Laguna bundles expose `tokenizer_config.json.chat_template`
    /// as `{% include 'chat_template.jinja' %}`. Some Jinja bridges can
    /// resolve that sidecar, but others either throw or render without
    /// directory context. Treat this as a production template, not a
    /// recovery path, so all hosts get the native Poolside turn format:
    /// `<system>`, `<user>`, `<assistant>`, `</think>` for direct-answer
    /// mode, and `<think>` for reasoning mode.
    public static let lagunaMinimal: String = #"""
{{- "〈|EOS|〉" -}}
{%- set enable_thinking = enable_thinking | default(false) -%}
{%- set add_generation_prompt = add_generation_prompt | default(false) -%}
{%- set system_message = "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users through natural language conversations." -%}
{%- if messages and messages[0].role == "system" -%}
  {%- set system_message = messages[0].content -%}
{%- endif -%}
{%- if (system_message and system_message.strip()) or tools -%}
  {{- "<system>\n" -}}
  {%- if system_message and system_message.strip() -%}
    {{- "\n" -}}{{- system_message.rstrip() -}}
  {%- endif -%}
  {%- if tools -%}
    {{- "\n\n### Tools\n\n" -}}
    {{- "You may call functions to assist with the user query.\n" -}}
    {{- "All available function signatures are listed below:\n" -}}
    {{- "<available_tools>\n" -}}
    {%- for tool in tools -%}
      {{- tool | tojson -}}{{- "\n" -}}
    {%- endfor -%}
    {{- "</available_tools>\n" -}}
  {%- endif -%}
  {{- "\n</system>\n" -}}
{%- endif -%}
{%- for message in messages -%}
  {%- if message.role == "system" -%}
    {#- handled above -#}
  {%- elif message.role == "user" -%}
    {{- "<user>\n" -}}
    {%- if message.content is string -%}
      {{- message.content -}}
    {%- else -%}
      {%- for item in message.content -%}
        {%- if item.type == "text" -%}{{- item.text -}}{%- endif -%}
      {%- endfor -%}
    {%- endif -%}
    {{- "\n</user>\n" -}}
  {%- elif message.role == "assistant" -%}
    {{- "<assistant>\n" -}}
    {%- set content = message.content if message.content is string else "" -%}
    {%- set reasoning_content = "" -%}
    {%- if message['reasoning'] is string -%}
      {%- set reasoning_content = message['reasoning'] -%}
    {%- elif message['reasoning_content'] is string -%}
      {%- set reasoning_content = message['reasoning_content'] -%}
    {%- endif -%}
    {%- if '</think>' in content -%}
      {%- if not reasoning_content -%}
        {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') -%}
      {%- endif -%}
      {%- set content = content.split('</think>')[-1].lstrip('\n') -%}
    {%- endif -%}
    {%- if reasoning_content -%}
      {{- "<think>\n" -}}{{- reasoning_content.strip() -}}{{- "\n</think>\n" -}}
    {%- else -%}
      {{- "</think>\n" -}}
    {%- endif -%}
    {%- if content.strip() -%}
      {{- content.strip() -}}
    {%- else -%}
      {%- for item in message.content -%}
        {%- if item.type == "text" -%}{{- item.text -}}{%- endif -%}
      {%- endfor -%}
    {%- endif -%}
    {{- "\n</assistant>\n" -}}
  {%- elif message.role == "tool" -%}
    {{- "<tool_response>\n" -}}
    {%- if message.content is string -%}
      {{- message.content -}}
    {%- else -%}
      {{- message.content | tojson -}}
    {%- endif -%}
    {{- "\n</tool_response>\n" -}}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
  {{- "<assistant>\n" -}}
  {%- if enable_thinking -%}
    {{- "<think>\n" -}}
  {%- else -%}
    {{- "</think>\n" -}}
  {%- endif -%}
{%- endif -%}
"""#

    /// Mistral 3 / Mistral 3.5 / Mistral-Medium-3.5 minimal fallback.
    ///
    /// The native template uses
    /// `{%- for message in loop_messages + [{'role':'__sentinel__'}] %}`
    /// — swift-jinja can't parse `+` concatenation inside the
    /// for-iterable expression, throwing `Expected '%}' after for
    /// loop.. Got plus instead`. Reported on real
    /// `Mistral-Medium-3.5-128B-JANGTQ` 2026-05-01.
    ///
    /// This fallback preserves the Mistral-family markers
    /// `[SYSTEM_PROMPT] / [INST] / [/INST] / [AVAILABLE_TOOLS] /
    /// [/AVAILABLE_TOOLS] / [TOOL_CALLS] / [TOOL_RESULTS]`. Drops
    /// the consecutive-message aggregation logic the native template
    /// uses (which is what needed the `+ [sentinel]` trick) — most
    /// chats don't have consecutive same-role messages.
    ///
    /// Bridge sniff: `convertTokenToId("[INST]") != nil` is the
    /// cleanest Mistral-family signal — it's a special token across
    /// every Mistral 3 / 3.5 / 4 distribution.
    /// REMOVED 2026-05-01 — same rationale as the Laguna deprecation
    /// directly above. The osaurus-ai/swift-jinja fork (58d21aa5)
    /// fixes the for-loop-iterable parser to accept binary `+`
    /// expressions, so Mistral 3.5's native template
    /// (`{%- for message in loop_messages + [{...}] %}`) now parses
    /// and renders correctly with the FULL `[MODEL_SETTINGS]
    /// {"reasoning_effort": "..."}` plumbing the model was trained on.
    /// Kept private as `_mistral3Minimal_DEPRECATED` for one release
    /// as defensive reference.
    private static let _mistral3Minimal_DEPRECATED: String = #"""
{%- if messages and messages[0].role == "system" -%}
  {{- "[SYSTEM_PROMPT]" -}}
  {%- if messages[0].content is string -%}
    {{- messages[0].content -}}
  {%- else -%}
    {%- for item in messages[0].content -%}
      {%- if item.type == "text" -%}{{- item.text -}}{%- endif -%}
    {%- endfor -%}
  {%- endif -%}
  {{- "[/SYSTEM_PROMPT]" -}}
{%- endif -%}
{%- set effort = reasoning_effort | default("none") -%}
{{- "[MODEL_SETTINGS]" -}}
{{- '{"reasoning_effort": "' -}}{{- effort -}}{{- '"}' -}}
{{- "[/MODEL_SETTINGS]" -}}
{%- if tools -%}
  {{- "[AVAILABLE_TOOLS]" -}}
  {{- tools | tojson -}}
  {{- "[/AVAILABLE_TOOLS]" -}}
{%- endif -%}
{%- for message in messages -%}
  {%- if message.role == "system" -%}
    {#- handled above -#}
  {%- elif message.role == "user" -%}
    {{- "[INST]" -}}
    {%- if message.content is string -%}
      {{- message.content -}}
    {%- else -%}
      {%- for item in message.content -%}
        {%- if item.type == "text" -%}{{- item.text -}}{%- endif -%}
        {%- if item.type == "image" -%}{{- "[IMG]" -}}{%- endif -%}
      {%- endfor -%}
    {%- endif -%}
    {{- "[/INST]" -}}
  {%- elif message.role == "assistant" -%}
    {%- if message.tool_calls -%}
      {{- "[TOOL_CALLS]" -}}
      {{- message.tool_calls | tojson -}}
      {{- "[/TOOL_CALLS]" -}}
    {%- else -%}
      {%- if message.content is string -%}
        {{- message.content -}}
      {%- else -%}
        {%- for item in message.content -%}
          {%- if item.type == "text" -%}{{- item.text -}}{%- endif -%}
        {%- endfor -%}
      {%- endif -%}
      {{- eos_token -}}
    {%- endif -%}
  {%- elif message.role == "tool" -%}
    {{- "[TOOL_RESULTS]" -}}
    {%- if message.content is string -%}
      {{- message.content -}}
    {%- else -%}
      {{- message.content | tojson -}}
    {%- endif -%}
    {{- "[/TOOL_RESULTS]" -}}
  {%- endif -%}
{%- endfor -%}
"""#

    /// MiniMax-M2 minimal chat template that honors `enable_thinking`.
    /// The bundle-shipped native template ignores the flag and always
    /// prefills `<think>\n` at the assistant tail. That makes the model
    /// emit reasoning forever in thinking-off chat workloads (and break
    /// loop detection because the parser routes everything to
    /// `Generation.reasoning` when no `</think>` ever arrives).
    /// This template mirrors the native structure but gates the
    /// trailing prefill on `enable_thinking`. When the flag is false we
    /// emit a closed empty block (`<think>\n</think>\n\n`) so the model
    /// produces direct content. When the flag is true (or unset) the
    /// behaviour matches the native template.
    public static let minimaxM2Minimal: String = #"""
{# MiniMax-M2 minimal chat template with enable_thinking honored.
   Mirrors the bundle-shipped native template structurally, but the
   trailing assistant prefill is gated on `enable_thinking` so callers
   that explicitly opt out of CoT get a clean direct-answer path.
   When enable_thinking is unset, defaults to true to preserve the
   model's training-time bias. #}
{%- set toolcall_begin_token = '<minimax:tool_call>' -%}
{%- set toolcall_end_token   = '</minimax:tool_call>' -%}
{%- set _enable_thinking = enable_thinking | default(true) -%}

{%- macro render_tool_namespace(namespace_name, tool_list) -%}
{%- for tool in tool_list -%}
<tool>{{ tool.function | tojson(ensure_ascii=False) }}</tool>
{% endfor -%}
{%- endmacro -%}

{%- macro visible_text(content) -%}
    {%- if content is string -%}{{ content }}
    {%- elif content is iterable and content is not mapping -%}
        {%- for item in content -%}
            {%- if item is mapping and item.type == 'text' -%}{{- item.text }}
            {%- elif item is string -%}{{- item }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}{{- content }}
    {%- endif -%}
{%- endmacro -%}

{%- macro build_system_message(system_message) -%}
    {%- if system_message and system_message.content -%}
        {{- visible_text(system_message.content) }}
    {%- else -%}
        {%- if model_identity is not defined -%}
            {%- set model_identity = "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax." -%}
        {%- endif -%}
        {{- model_identity }}
    {%- endif -%}
{%- endmacro -%}

{%- set system_message = none -%}
{%- set conversation_messages = messages -%}
{%- if messages and messages[0].role == "system" -%}
    {%- set system_message = messages[0] -%}
    {%- set conversation_messages = messages[1:] -%}
{%- endif -%}

{%- set ns = namespace(last_user_index=-1) %}
{% for m in conversation_messages %}
    {%- if m.role == 'user' %}
        {% set ns.last_user_index = loop.index0 -%}
    {%- endif %}
{%- endfor %}

{{- ']~!b[' ~ ']~b]system' ~ '\n' }}
{{- build_system_message(system_message) }}

{%- if tools -%}
    {{- '\n\n' ~ '# Tools' ~ '\n' ~ 'You may call one or more tools to assist with the user query.\nHere are the tools available in JSONSchema format:' ~ '\n' }}
    {{- '\n' ~ '<tools>' ~ '\n' }}
    {{- render_tool_namespace("functions", tools) }}
    {{- '</tools>' ~ '\n\n' }}
{{- 'When making tool calls, use XML format to invoke tools and pass parameters:' ~ '\n' }}
{{- '\n' ~ toolcall_begin_token }}
<invoke name="tool-name-1">
<parameter name="param-key-1">param-value-1</parameter>
<parameter name="param-key-2">param-value-2</parameter>
...
</invoke>
{{- '\n' ~ toolcall_end_token }}
{%- endif -%}
{{- '[e~[\n' }}

{%- set last_tool_call = namespace(name=none) -%}
{%- for message in conversation_messages -%}
    {%- if message.role == 'assistant' -%}
        {{- ']~b]ai' ~ '\n' }}
        {%- set reasoning_content = '' %}
        {%- set content = visible_text(message.content) %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].strip('\n').split('<think>')[-1].strip('\n') %}
                {%- set content = content.split('</think>')[-1].strip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if reasoning_content and loop.index0 > ns.last_user_index -%}
            {{- '<think>' ~ '\n' ~ reasoning_content ~ '\n' ~ '</think>' ~ '\n\n' }}
        {%- endif -%}
        {%- if content -%}{{- content }}{%- endif -%}
        {%- if message.tool_calls -%}
            {{- '\n' ~ toolcall_begin_token ~ '\n' }}
            {%- for tool_call in message.tool_calls -%}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<invoke name="' + tool_call.name + '">' }}
                {% set _args = tool_call.arguments %}
                {%- for k, v in _args.items() %}
                {{- '<parameter name="' + k + '">' }}
                {{- v | tojson(ensure_ascii=False) if v is not string else v }}
                {{- '</parameter>' }}
                {% endfor %}
                {{- '</invoke>' ~ '\n' }}
            {%- endfor -%}
            {{- toolcall_end_token}}
            {%- set last_tool_call.name = message.tool_calls[-1].name -%}
        {%- else -%}
            {%- set last_tool_call.name = none -%}
        {%- endif -%}
        {{- '[e~[' ~ '\n' }}
    {%- elif message.role == 'tool' -%}
    {%- if last_tool_call.name is none -%}
        {{- raise_exception("Message has tool role, but there was no previous assistant message with a tool call!") }}
    {%- endif -%}
    {%- if loop.first or (conversation_messages[loop.index0 - 1].role != 'tool') -%}
        {{- ']~b]tool' }}
    {%- endif -%}
    {%- if message.content is string -%}
        {{- '\n<response>' }}
        {{- message.content }}
        {{- '</response>' }}
    {%- else -%}
        {%- for tr in message.content -%}
            {{- '\n<response>' }}
            {{- tr.output if tr.output is defined else (tr.text if tr.type == 'text' and tr.text is defined else tr) }}
            {{- '\n</response>' }}
        {%- endfor -%}
    {%- endif -%}
    {%- if loop.last or (conversation_messages[loop.index0 + 1].role != 'tool') -%}
        {{- '[e~[\n' -}}
    {%- endif -%}
    {%- elif message.role == 'user' -%}
        {{- ']~b]user' ~ '\n' }}
        {{- visible_text(message.content) }}
        {{- '[e~[' ~ '\n' }}
    {%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt -%}
    {{- ']~b]ai' ~ '\n' }}
    {%- if _enable_thinking -%}
        {{- '<think>' ~ '\n' }}
    {%- else -%}
        {{- '<think>\n</think>\n\n' }}
    {%- endif -%}
{%- endif -%}
"""#

    /// ZAYA1-VL multimodal template with the same Zyphra XML tool
    /// declaration/call contract as text ZAYA, while preserving the shipped
    /// vision placeholder markers.
    ///
    /// ZAYA1-VL bundles stamp `think_in_template=false`, so this fallback must
    /// not prefill `<think>` or `<think></think>` rails. The runtime reasoning
    /// parser still handles real model-emitted `<think>...</think>` blocks, but
    /// the template does not manufacture them.
    public static let zayaVLVisionToolMinimal: String = #"""
{{- bos_token -}}
{%- if tools is not defined -%}
    {%- set tools = [] -%}
{%- endif -%}
{%- set required_tool_choice = false -%}
{%- set required_tool_name = '' -%}
{%- if tool_choice is defined and tool_choice == 'required' -%}
    {%- set required_tool_choice = true -%}
{%- elif additionalContext is defined and additionalContext['tool_choice'] == 'required' -%}
    {%- set required_tool_choice = true -%}
{%- endif -%}
{%- if tool_choice_name is defined -%}
    {%- set required_tool_name = tool_choice_name -%}
{%- elif additionalContext is defined and additionalContext['tool_choice_name'] is defined -%}
    {%- set required_tool_name = additionalContext['tool_choice_name'] -%}
{%- endif -%}
{%- if required_tool_choice and not required_tool_name and tools is iterable and tools | length == 1 -%}
    {%- set only_required_tool = tools[0]['function'] if tools[0]['function'] is defined else tools[0] -%}
    {%- if only_required_tool['name'] is defined -%}
        {%- set required_tool_name = only_required_tool['name'] -%}
    {%- endif -%}
{%- endif -%}

{%- macro render_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- elif content is sequence and content is not mapping -%}
        {%- for item in content -%}
            {%- if item is mapping and item['type'] == 'image' -%}
                {{- '<|vision_start|><image><|vision_end|>\n' -}}
            {%- elif item is mapping and item['type'] == 'text' -%}
                {{- item['text'] -}}
            {%- elif item is string -%}
                {{- item -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}

{%- macro render_text_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- elif content is sequence and content is not mapping -%}
        {%- for item in content -%}
            {%- if item is mapping and item['type'] == 'text' -%}
                {{- item['text'] -}}
            {%- elif item is string -%}
                {{- item -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}

{%- macro render_tool_call(tool_call) -%}
    {%- if tool_call['function'] is defined -%}
        {%- set tool_call = tool_call['function'] -%}
    {%- endif -%}
    {{- '<zyphra_tool_call>\n<function=' ~ tool_call['name'] ~ '>\n' -}}
    {%- if tool_call['arguments'] is defined -%}
        {%- for args_name, args_value in tool_call['arguments']|items -%}
            {{- '<parameter=' ~ args_name ~ '>\n' -}}
            {{- (args_value | tojson | safe) if args_value is mapping or (args_value is sequence and args_value is not string) else (args_value | string) -}}
            {{- '\n</parameter>\n' -}}
        {%- endfor -%}
    {%- endif -%}
    {{- '</function>\n</zyphra_tool_call>\n' -}}
{%- endmacro -%}

{%- macro render_required_tool_choice_instruction(latest_user_content='') -%}
    {%- set latest_user_text = render_text_content(latest_user_content) -%}
    {{- '<IMPORTANT>\nThe current assistant response MUST be a tool call. Reply only with a `<zyphra_tool_call>` block for one available function and no prose before the tool result. Include every required `<parameter=...>` value exactly as requested.' -}}
    {%- if required_tool_name -%}
        {{- '\nUse the `' ~ required_tool_name ~ '` function.' -}}
        {%- for tool in tools -%}
            {%- set selected_tool = tool['function'] if tool['function'] is defined else tool -%}
            {%- if selected_tool['name'] == required_tool_name and selected_tool['parameters'] is defined and selected_tool['parameters']['required'] is defined -%}
                {{- '\nRequired parameters for `' ~ required_tool_name ~ '`: ' ~ (selected_tool['parameters']['required'] | join(', ')) ~ '.' -}}
                {%- if latest_user_text is string and ('exact' in latest_user_text or 'preserving newlines:' in latest_user_text) -%}
                    {{- '\nRequired call shape for the current request:\n<zyphra_tool_call>\n<function=' ~ required_tool_name ~ '>' -}}
                    {%- for param_name in selected_tool['parameters']['required'] -%}
                        {%- set exact = namespace(value='') -%}
                        {%- set exact_markers = [
                            'on this exact ' ~ param_name ~ ':',
                            'exact ' ~ param_name ~ ':',
                            'this exact ' ~ param_name ~ ':',
                            'exactly this ' ~ param_name ~ ':',
                            'exactly ' ~ param_name ~ ':',
                            'on this exact text:',
                            'this exact text:',
                            'exact text:',
                            'on exactly this text:',
                            'exactly this text:',
                            'exactly this new ' ~ param_name ~ ', preserving newlines:',
                            'exactly this ' ~ param_name ~ ', preserving newlines:',
                            'this new ' ~ param_name ~ ', preserving newlines:',
                            'new ' ~ param_name ~ ', preserving newlines:',
                            param_name ~ ', preserving newlines:',
                            'preserving newlines:'
                        ] -%}
                        {%- for exact_marker in exact_markers -%}
                            {%- if not exact.value and exact_marker in latest_user_text -%}
                                {%- set exact.value = latest_user_text.split(exact_marker)[1] | trim -%}
                            {%- endif -%}
                        {%- endfor -%}
                        {{- '\n<parameter=' ~ param_name ~ '>\n' -}}
                        {%- if exact.value -%}
                            {{- exact.value -}}
                        {%- endif -%}
                        {{- '\n</parameter>' -}}
                    {%- endfor -%}
                    {{- '\n</function>\n</zyphra_tool_call>' -}}
                {%- endif -%}
                {{- '\nDo not omit required parameters. If the latest user message asks to use the tool on exact text, copy that exact text into the string parameter body, preserving newlines.' -}}
                {{- '\nFor string parameters, write the raw string value only. Do not wrap the parameter value in JSON quotes unless the requested value itself includes quote characters.' -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
    {{- '\n</IMPORTANT>' -}}
{%- endmacro -%}

{%- set loop_messages = messages -%}
{%- set has_system = (messages | length > 0 and messages[0]['role'] == 'system') -%}
{%- if has_system or (tools is iterable and tools | length > 0) -%}
    {{- '<|im_start|>system\n' -}}
    {%- if has_system -%}
        {{- render_content(messages[0]['content']) -}}
        {%- set loop_messages = messages[1:] -%}
    {%- endif -%}
    {%- if tools is iterable and tools | length > 0 -%}
        {%- if has_system -%}{{- '\n\n' -}}{%- endif -%}
        {{- '# Tools\n\nYou have access to the following functions:\n\n<tools>' -}}
        {%- for tool in tools -%}
            {%- if tool['function'] is defined -%}
                {%- set tool = tool['function'] -%}
            {%- endif -%}
            {{- '\n<function>\n<name>' ~ tool['name'] ~ '</name>' -}}
            {%- if tool['description'] is defined -%}
                {{- '\n<description>' ~ (tool['description'] | trim) ~ '</description>' -}}
            {%- endif -%}
            {%- if tool['parameters'] is defined and tool['parameters']['properties'] is defined -%}
                {{- '\n<parameters>' -}}
                {%- for param_name, param in tool['parameters']['properties'] | dictsort -%}
                    {{- '\n<parameter>\n<name>' ~ param_name ~ '</name>' -}}
                    {%- if param['type'] is defined -%}
                        {{- '\n<type>' ~ param['type'] ~ '</type>' -}}
                    {%- endif -%}
                    {%- if param['description'] is defined -%}
                        {{- '\n<description>' ~ (param['description'] | trim) ~ '</description>' -}}
                    {%- endif -%}
                    {{- '\n</parameter>' -}}
                {%- endfor -%}
                {%- if tool['parameters']['required'] is defined -%}
                    {{- '\n<required>' ~ (tool['parameters']['required'] | tojson | safe) ~ '</required>' -}}
                {%- endif -%}
                {{- '\n</parameters>' -}}
            {%- elif tool['parameters'] is defined -%}
                {{- '\n<parameters>' ~ (tool['parameters'] | tojson | safe) ~ '</parameters>' -}}
            {%- endif -%}
            {{- '\n</function>' -}}
        {%- endfor -%}
        {%- if required_tool_choice -%}
            {{- '\n</tools>\n\nWhen the current assistant response is a function call, reply with one `<zyphra_tool_call>` block matching one listed function and no prose.' -}}
        {%- else -%}
            {{- '\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<zyphra_tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</zyphra_tool_call>' -}}
        {%- endif -%}
        {%- if required_tool_choice -%}
            {{- '\n\n' -}}
            {{- render_required_tool_choice_instruction() -}}
        {%- endif -%}
    {%- endif -%}
    {{- '<|im_end|>\n' -}}
{%- endif -%}

{%- for message in loop_messages -%}
    {%- if message['role'] == 'user' or message['role'] == 'question' -%}
        {%- set next_is_pure_tool_call = false -%}
        {%- if required_tool_choice and not loop.last -%}
            {%- set next_message = loop_messages[loop.index0 + 1] -%}
            {%- set next_has_tool_calls = next_message['tool_calls'] is defined and next_message['tool_calls'] is iterable and next_message['tool_calls'] | length > 0 -%}
            {%- set next_assistant_content = render_content(next_message['content']) -%}
            {%- if next_message['role'] == 'assistant' and next_has_tool_calls and not next_assistant_content -%}
                {%- set next_is_pure_tool_call = true -%}
            {%- endif -%}
        {%- endif -%}
        {%- if (not required_tool_choice or loop.last) and not next_is_pure_tool_call -%}
        {{- '<|im_start|>user\n' -}}
        {{- render_content(message['content']) -}}
        {%- if required_tool_choice and loop.last -%}
            {{- '\n\n' -}}
            {{- render_required_tool_choice_instruction(message['content']) -}}
        {%- endif -%}
        {{- '<|im_end|>\n' -}}
        {%- endif -%}
    {%- elif message['role'] == 'assistant' -%}
        {%- set has_tool_calls = message['tool_calls'] is defined and message['tool_calls'] is iterable and message['tool_calls'] | length > 0 -%}
        {%- set assistant_content = render_content(message['content']) -%}
        {%- if not required_tool_choice and not (required_tool_choice and has_tool_calls and not assistant_content) -%}
        {{- '<|im_start|>assistant\n' -}}
        {{- assistant_content -}}
        {%- if has_tool_calls and not required_tool_choice -%}
            {%- for tool_call in message['tool_calls'] -%}
                {{- render_tool_call(tool_call) -}}
            {%- endfor -%}
        {%- endif -%}
        {{- '<|im_end|>\n' -}}
        {%- endif -%}
    {%- elif message['role'] == 'tool' -%}
        {%- if not required_tool_choice -%}
            {{- '<|im_start|>user\n<zyphra_tool_response>\n' -}}
            {{- render_content(message['content']) -}}
            {{- '\n</zyphra_tool_response>\n<|im_end|>\n' -}}
        {%- endif -%}
    {%- elif message['role'] == 'system' -%}
        {{- '<|im_start|>system\n' -}}
        {{- render_content(message['content']) -}}
        {{- '<|im_end|>\n' -}}
    {%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\n' -}}
{%- endif -%}
"""#

    /// LFM2/LFM2.5 text template with Liquid's native Python-call tool
    /// syntax:
    /// `<|tool_call_start|>[tool_name(arg='value')]<|tool_call_end|>`.
    ///
    /// Some LFM2.5 JANG bundles ship a native sidecar that lists tools and
    /// can replay historical assistant tool calls, but does not consume the
    /// OpenAI-compatible `tool_choice` fields. This fallback keeps the same
    /// ChatML-ish role markers and LFM tool-call syntax while adding only the
    /// explicit required/named tool-choice contract. Optional tools remain
    /// optional, and the assistant tail does not synthesize `<think>`.
    public static let lfm2ToolMinimal: String = #"""
{{- bos_token -}}
{%- set required_tool_choice = false -%}
{%- set required_tool_name = '' -%}
{%- if tool_choice is defined and tool_choice == 'required' -%}
    {%- set required_tool_choice = true -%}
{%- elif additionalContext is defined and additionalContext['tool_choice'] == 'required' -%}
    {%- set required_tool_choice = true -%}
{%- endif -%}
{%- if tool_choice_name is defined -%}
    {%- set required_tool_name = tool_choice_name -%}
{%- elif additionalContext is defined and additionalContext['tool_choice_name'] is defined -%}
    {%- set required_tool_name = additionalContext['tool_choice_name'] -%}
{%- endif -%}
{%- if required_tool_choice and not required_tool_name and tools is iterable and tools | length == 1 -%}
    {%- set only_required_tool = tools[0]['function'] if tools[0]['function'] is defined else tools[0] -%}
    {%- if only_required_tool['name'] is defined -%}
        {%- set required_tool_name = only_required_tool['name'] -%}
    {%- endif -%}
{%- endif -%}

{%- macro format_arg_value(arg_value) -%}
    {%- if arg_value is string -%}
        {{- "'" + arg_value + "'" -}}
    {%- elif arg_value is mapping or (arg_value is sequence and arg_value is not string) -%}
        {{- arg_value | tojson -}}
    {%- else -%}
        {{- arg_value | string -}}
    {%- endif -%}
{%- endmacro -%}

{%- macro parse_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- elif content is sequence and content is not mapping -%}
        {%- for item in content -%}
            {%- if item is mapping and item['type'] == 'text' -%}
                {{- item['text'] -}}
            {%- elif item is mapping and item['type'] == 'image' -%}
                {{- '<image>' -}}
            {%- elif item is string -%}
                {{- item -}}
            {%- else -%}
                {{- item | tojson -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}

{%- macro render_tool_calls(tool_calls) -%}
    {%- set tool_calls_ns = namespace(tool_calls=[]) -%}
    {%- for tool_call in tool_calls -%}
        {%- set fn = tool_call['function'] if tool_call['function'] is defined else tool_call -%}
        {%- set args = fn['arguments'] if fn['arguments'] is defined else {} -%}
        {%- set args_ns = namespace(arg_strings=[]) -%}
        {%- if args is mapping -%}
            {%- for arg_name, arg_value in args | dictsort -%}
                {%- set args_ns.arg_strings = args_ns.arg_strings + [arg_name + '=' + format_arg_value(arg_value)] -%}
            {%- endfor -%}
        {%- endif -%}
        {%- set tool_calls_ns.tool_calls = tool_calls_ns.tool_calls + [fn['name'] + '(' + (args_ns.arg_strings | join(', ')) + ')'] -%}
    {%- endfor -%}
    {{- '<|tool_call_start|>[' + (tool_calls_ns.tool_calls | join(', ')) + ']<|tool_call_end|>' -}}
{%- endmacro -%}

{%- macro render_required_tool_choice_instruction(latest_user_content='') -%}
    {{- 'The active API tool_choice is required for this assistant turn. Reply only with one native LFM tool call and no prose before the tool result.' -}}
    {%- if required_tool_name -%}
        {{- '\nUse the `' ~ required_tool_name ~ '` function.' -}}
        {%- for tool in tools -%}
            {%- set selected_tool = tool['function'] if tool['function'] is defined else tool -%}
            {%- if selected_tool['name'] == required_tool_name and selected_tool['parameters'] is defined and selected_tool['parameters']['required'] is defined -%}
                {{- '\nRequired parameters for `' ~ required_tool_name ~ '`: ' ~ (selected_tool['parameters']['required'] | join(', ')) ~ '.' -}}
                {%- for param_name in selected_tool['parameters']['required'] -%}
                    {%- set exact = namespace(value='') -%}
                    {%- set latest_user_text = parse_content(latest_user_content) -%}
                    {%- set exact_markers = [
                        'on this exact ' ~ param_name ~ ':',
                        'this exact ' ~ param_name ~ ':',
                        'exact ' ~ param_name ~ ':',
                        'on exactly this ' ~ param_name ~ ':',
                        'exactly this ' ~ param_name ~ ':',
                        'on this exact text:',
                        'this exact text:',
                        'exact text:',
                        'on exactly this text:',
                        'exactly this text:',
                        'use ' ~ required_tool_name ~ ' on this exact text:',
                        'now use ' ~ required_tool_name ~ ' on this exact text:'
                    ] -%}
                    {%- for marker in exact_markers -%}
                        {%- if not exact.value and latest_user_text is string and marker in latest_user_text -%}
                            {%- set exact.value = latest_user_text.split(marker)[1] | trim -%}
                        {%- endif -%}
                    {%- endfor -%}
                    {%- if exact.value -%}
                        {{- '\nRequired call shape for the current request:\n<|tool_call_start|>[' ~ required_tool_name ~ '(' ~ param_name ~ '=' ~ "'" ~ exact.value ~ "'" ~ ')]<|tool_call_end|>' -}}
                    {%- endif -%}
                {%- endfor -%}
                {{- '\nDo not omit required parameters. If the latest user message asks to use exact text, copy that exact text into the string argument, preserving every newline and adding no spaces.' -}}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{- '\nUse exactly one of the listed function names and include every required argument.' -}}
    {%- endif -%}
{%- endmacro -%}

{%- set ns = namespace(system_prompt='', last_user_index=-1) -%}
{%- set loop_messages = messages -%}
{%- if messages and messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is defined -%}
        {%- set ns.system_prompt = parse_content(messages[0]['content']) -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if message['role'] == 'user' -%}
        {%- set ns.last_user_index = loop.index0 -%}
    {%- endif -%}
{%- endfor -%}
{%- if ns.system_prompt or (tools is iterable and tools | length > 0) -%}
    {{- '<|im_start|>system\n' -}}
    {%- if ns.system_prompt -%}
        {{- ns.system_prompt -}}
    {%- endif -%}
    {%- if tools is iterable and tools | length > 0 -%}
        {%- if ns.system_prompt -%}{{- '\n\n' -}}{%- endif -%}
        {{- 'List of tools: ' ~ (tools | tojson) -}}
        {%- if required_tool_choice -%}
            {{- '\n\n' ~ render_required_tool_choice_instruction() -}}
        {%- endif -%}
    {%- endif -%}
    {{- '<|im_end|>\n' -}}
{%- endif -%}

{%- for message in loop_messages -%}
    {{- '<|im_start|>' + message['role'] + '\n' -}}
    {%- if message['role'] == 'assistant' -%}
        {%- if message['thinking'] is defined -%}
            {{- '<think>' + message['thinking'] + '</think>' -}}
        {%- endif -%}
        {%- if message['content'] is defined -%}
            {{- parse_content(message['content']) -}}
        {%- endif -%}
        {%- if message['tool_calls'] is defined and message['tool_calls'] -%}
            {{- render_tool_calls(message['tool_calls']) -}}
        {%- endif -%}
    {%- elif message['role'] == 'user' -%}
        {{- parse_content(message['content']) -}}
        {%- if required_tool_choice and loop.index0 == ns.last_user_index -%}
            {{- '\n\n' ~ render_required_tool_choice_instruction(message['content']) -}}
        {%- endif -%}
    {%- elif message['role'] == 'tool' -%}
        {{- parse_content(message['content']) -}}
    {%- else -%}
        {%- if message['content'] is defined -%}
            {{- parse_content(message['content']) -}}
        {%- endif -%}
    {%- endif -%}
    {{- '<|im_end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\n' -}}
{%- endif -%}
"""#

    /// Ordered list of (label, template) fallbacks used when the
    /// model's native template throws. Order matters: `gemma4WithTools`
    /// comes first because (a) it subsumes `gemma4Minimal` when no
    /// tools are present, and (b) Gemma-4 is the most common family
    /// blocked by the upstream parser bug.
    public static let orderedFallbacks: [(label: String, template: String)] = [
        ("Gemma4WithTools", gemma4WithTools),
        ("Gemma4Minimal",   gemma4Minimal),
        ("NemotronMinimal", nemotronMinimal),
        ("DSV4Minimal",     dsv4Minimal),
    ]
}
