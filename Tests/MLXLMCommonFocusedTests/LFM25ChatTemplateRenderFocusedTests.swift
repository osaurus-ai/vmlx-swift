// Copyright 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLXLMCommon
import Testing
import VMLXJinja

/// Renders the exact LFM2.5-2.6B bundle chat template (identical bytes in
/// the JANG_6M and MXFP8 bundles, and in upstream LiquidAI's release) and
/// pins the constructs the runtime depends on:
///
/// - `add_generation_prompt` ALWAYS ends the prompt inside an open
///   `<think>` block — this model has no `enable_thinking` knob, so the
///   reasoning parser must start in-reasoning from the prompt tail;
/// - assistant tool calls render as the Pythonic
///   `<|tool_call_start|>[fn(a='x')]<|tool_call_end|>` envelope that
///   `LFM2ToolCallParser` parses back;
/// - prior-turn `<think>` bodies are stripped from history before the last
///   user turn (`preserve_thinking` defaults false via `| default(false)`);
/// - the `{% generation %}`/`{% endgeneration %}` tags, `.get()`,
///   `.endswith()`, `.split()[-1]`, and slice constructs all execute.
@Suite("LFM2.5-2.6B chat template render")
struct LFM25ChatTemplateRenderFocusedTests {

    static let lfm25Template = #"""
        {{- bos_token -}}
        {%- set preserve_thinking = preserve_thinking | default(false) -%}

        {%- macro format_arg_value(arg_value) -%}
            {%- if arg_value is string -%}
                {{- "'" + (arg_value | replace("\\", "\\\\") | replace("'", "\\'") | replace("\n", "\\n") | replace("\r", "\\r")) + "'" -}}
            {%- elif arg_value is mapping or arg_value is iterable -%}
                {{- arg_value | tojson -}}
            {%- else -%}
                {{- arg_value | string -}}
            {%- endif -%}
        {%- endmacro -%}

        {%- macro parse_content(content) -%}
            {%- if content is string -%}
                {{- content -}}
            {%- elif content is mapping -%}
                {{- content | tojson -}}
            {%- elif content is iterable -%}
                {%- set _ns = namespace(result="") -%}
                {%- for item in content -%}
                    {%- if item is string -%}
                        {%- set _ns.result = _ns.result + item -%}
                    {%- elif item is mapping and item.get("type") == "image" -%}
                        {%- set _ns.result = _ns.result + "<image>" -%}
                    {%- elif item is mapping and item.get("type") == "text" -%}
                        {%- set _ns.result = _ns.result + ((item.get("text") or "") | string) -%}
                    {%- else -%}
                        {%- set _ns.result = _ns.result + (item | tojson) -%}
                    {%- endif -%}
                {%- endfor -%}
                {{- _ns.result -}}
            {%- endif -%}
        {%- endmacro -%}

        {%- macro render_tool_calls(tool_calls) -%}
            {%- set tool_calls_ns = namespace(tool_calls=[]) -%}
            {%- for tool_call in tool_calls -%}
                {%- set func = tool_call["function"] if "function" in tool_call else tool_call -%}
                {%- set func_name = func["name"] -%}
                {%- set func_args = func.get("arguments") -%}
                {%- set args_ns = namespace(arg_strings=[]) -%}
                {%- if func_args is mapping -%}
                    {%- for arg_name, arg_value in func_args.items() -%}
                        {%- set args_ns.arg_strings = args_ns.arg_strings + [arg_name + "=" + format_arg_value(arg_value)] -%}
                    {%- endfor -%}
                {%- elif func_args is string and (func_args | trim) not in ["", "{}", "null"] -%}
                    {{- raise_exception("Tool call arguments must be a mapping, got a JSON-encoded string: parse arguments with json.loads() before applying the chat template") -}}
                {%- endif -%}
                {%- set tool_calls_ns.tool_calls = tool_calls_ns.tool_calls + [func_name + "(" + (args_ns.arg_strings | join(", ")) + ")"] -%}
            {%- endfor -%}
            {{- "<|tool_call_start|>[" + (tool_calls_ns.tool_calls | join(", ")) + "]<|tool_call_end|>" -}}
        {%- endmacro -%}

        {%- set ns = namespace(system_prompt="", last_user_index=-1) -%}
        {%- if messages and messages[0]["role"] == "system" -%}
            {%- if messages[0].get("content") -%}
                {%- set ns.system_prompt = parse_content(messages[0]["content"]) -%}
            {%- endif -%}
            {%- set messages = messages[1:] -%}
        {%- endif -%}
        {%- if tools -%}
            {%- set ns.system_prompt = ns.system_prompt + ("\n" if ns.system_prompt else "") + "List of tools: [" -%}
            {%- for tool in tools -%}
                {%- if tool is not string -%}
                    {%- set tool = tool | tojson -%}
                {%- endif -%}
                {%- set ns.system_prompt = ns.system_prompt + tool -%}
                {%- if not loop.last -%}
                    {%- set ns.system_prompt = ns.system_prompt + ", " -%}
                {%- endif -%}
            {%- endfor -%}
            {%- set ns.system_prompt = ns.system_prompt + "]" -%}
        {%- endif -%}
        {%- if ns.system_prompt -%}
            {{- "<|im_start|>system\n" + ns.system_prompt + "<|im_end|>\n" -}}
        {%- endif -%}
        {%- for message in messages -%}
            {%- if message["role"] == "user" -%}
                {%- set ns.last_user_index = loop.index0 -%}
            {%- endif -%}
        {%- endfor -%}
        {%- for message in messages -%}
            {{- "<|im_start|>" + message.role + "\n" -}}
            {%- if message.role == "assistant" -%}
                {%- generation -%}
                {%- set keep_thinking = preserve_thinking or loop.index0 > ns.last_user_index -%}
                {%- set thinking = message.thinking or message.reasoning or message.reasoning_content -%}
                {%- set thinking = thinking if thinking is string else "" -%}
                {%- if thinking and keep_thinking -%}
                    {{- "<think>" + thinking + "</think>" -}}
                {%- endif -%}
                {%- set _cfm_tag = "CONTINUE_FINAL_MESSAGE_TAG " -%}
                {%- set _has_cfm = false -%}
                {%- set content = "" -%}
                {%- if message.get("content") -%}
                    {%- set content = parse_content(message.content) -%}
                {%- endif -%}
                {%- if not keep_thinking and "</think>" in content -%}
                    {%- set content = content.split("</think>")[-1] | trim -%}
                {%- endif -%}
                {%- if content.endswith(_cfm_tag) -%}
                    {%- set _has_cfm = true -%}
                    {%- set _trunc_len = (content | length) - (_cfm_tag | length) -%}
                    {%- set content = content[:_trunc_len] -%}
                {%- endif -%}
                {{- content -}}
                {%- if message.tool_calls -%}
                    {{- render_tool_calls(message.tool_calls) -}}
                {%- endif -%}
                {%- if _has_cfm -%}
                    {{- _cfm_tag -}}
                {%- endif -%}
                {{- "<|im_end|>\n" -}}
                {%- endgeneration -%}
            {%- else %}
                {%- if message.get("content") -%}
                    {{- parse_content(message["content"]) -}}
                {%- endif -%}
                {{- "<|im_end|>\n" -}}
            {%- endif %}
        {%- endfor -%}
        {%- if add_generation_prompt -%}
            {{- "<|im_start|>assistant\n<think>" -}}
        {%- endif -%}
        """#

    private func render(_ context: [String: any Sendable]) throws -> String {
        let template = try Template(Self.lfm25Template)
        var values: [String: Value] = [:]
        for (key, value) in context {
            values[key] = try Value(any: value)
        }
        if values["bos_token"] == nil {
            values["bos_token"] = try Value(any: "<|startoftext|>")
        }
        return try template.render(values)
    }

    @Test("generation prompt always ends inside an open think block")
    func generationPromptOpensThink() throws {
        let out = try render([
            "messages": [
                ["role": "system", "content": "You are helpful."],
                ["role": "user", "content": "Hi"],
            ],
            "add_generation_prompt": true,
        ])
        #expect(out.hasPrefix("<|startoftext|><|im_start|>system\nYou are helpful.<|im_end|>\n"))
        #expect(out.hasSuffix("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>"))
    }

    @Test("tools land in the system prompt as a JSON tool list")
    func toolsRenderIntoSystemPrompt() throws {
        let out = try render([
            "messages": [["role": "user", "content": "What time is it?"]],
            "tools": [
                [
                    "type": "function",
                    "function": [
                        "name": "get_current_time",
                        "parameters": ["type": "object", "properties": [String: any Sendable]()]
                            as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable]
            ],
            "add_generation_prompt": true,
        ])
        #expect(out.contains("<|im_start|>system\nList of tools: ["))
        #expect(out.contains("get_current_time"))
        #expect(out.hasSuffix("<|im_start|>assistant\n<think>"))
    }

    @Test("assistant tool calls round-trip through the Pythonic envelope")
    func assistantToolCallsRenderPythonic() throws {
        let out = try render([
            "messages": [
                ["role": "user", "content": "Weather in Paris then Tokyo"],
                [
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        [
                            "function": [
                                "name": "get_weather",
                                "arguments": ["city": "Paris"],
                            ] as [String: any Sendable]
                        ] as [String: any Sendable]
                    ],
                ] as [String: any Sendable],
                ["role": "tool", "content": "18C sunny"],
                ["role": "user", "content": "And Tokyo?"],
            ],
            "add_generation_prompt": true,
        ])
        #expect(out.contains("<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>"))
        #expect(out.hasSuffix("<|im_start|>assistant\n<think>"))

        // The rendered envelope must be exactly what LFM2ToolCallParser
        // accepts back — the round-trip contract for multiturn tool history.
        let parser = LFM2ToolCallParser()
        let calls = parser.parseEOS(
            "<|tool_call_start|>[get_weather(city='Paris')]<|tool_call_end|>",
            tools: nil)
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "get_weather")
    }

    @Test("history thinking is stripped before the last user turn by default")
    func historyThinkingStripped() throws {
        let out = try render([
            "messages": [
                ["role": "user", "content": "First question"],
                [
                    "role": "assistant",
                    "content": "secret deliberation</think>The answer is 4.",
                ],
                ["role": "user", "content": "Second question"],
            ],
            "add_generation_prompt": true,
        ])
        #expect(!out.contains("secret deliberation"))
        #expect(out.contains("<|im_start|>assistant\nThe answer is 4.<|im_end|>\n"))
        #expect(out.hasSuffix("<|im_start|>user\nSecond question<|im_end|>\n<|im_start|>assistant\n<think>"))
    }
}
