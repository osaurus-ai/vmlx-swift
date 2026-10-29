"""MiniMax-M3.1 thinking_mode template fixture for MiniMaxM3ThinkingModeTests (2026-10-28).

Renders the M3.1 template with thinking_mode branches (chat_template.jinja, sha256 e8fa0fa5...) through HF
transformers' apply_chat_template for each conversation x request context, after the request -> template mapping the
Swift adapter implements (explicit thinking_mode in {enabled, disabled, adaptive} wins; else enable_thinking True ->
enabled, False -> disabled; else adaptive = unset; reasoning_effort passed through). Dict keys are inserted sorted
(recursively), which is the order swift-jinja's Value(any:) gives a Swift dictionary.

  python generate_thinking_mode_fixtures.py <tokenizer dir> <fixture dir>
"""
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


def srt(x):
    if isinstance(x, dict):
        return {k: srt(x[k]) for k in sorted(x)}
    if isinstance(x, list):
        return [srt(v) for v in x]
    return x


tok_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
tok = AutoTokenizer.from_pretrained(tok_dir)
new = (out / "chat_template.jinja").read_text()
shipped = (out / "chat_template.shipped.jinja").read_text()
tools = srt([{"type": "function", "function": {"name": "run_shell", "description": "Run a shell command",
              "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}])
convs = {
    "plain": [{"role": "user", "content": "What is two plus two?"}],
    "system": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}],
    "tool_loop": [
        {"role": "user", "content": "List files."},
        {"role": "assistant", "reasoning_content": "Use ls.", "content": "",
         "tool_calls": [{"type": "function", "function": {"name": "run_shell", "arguments": {"cmd": "ls -la"}}}]},
        {"role": "tool", "content": "a.txt\nb.txt"},
        {"role": "assistant", "content": "</mm:think>Two files: a.txt and b.txt."},
        {"role": "user", "content": "Now count lines in a.txt"}],
}
requests = [{}, {"enable_thinking": True}, {"enable_thinking": False}, {"thinking_mode": "enabled"},
            {"thinking_mode": "disabled"}, {"thinking_mode": "adaptive"}, {"thinking_mode": "auto"},
            {"thinking_mode": "instruct", "enable_thinking": True}, {"thinking_mode": "enabled", "enable_thinking": False},
            {"thinking_mode": "disabled", "enable_thinking": True}]


def mapped(req):
    mode = req.get("thinking_mode")
    if mode not in ("enabled", "disabled", "adaptive"):
        mode = {True: "enabled", False: "disabled"}.get(req.get("enable_thinking"))
    ctx = {}
    if mode in ("enabled", "disabled"):
        ctx["thinking_mode"] = mode
    if "reasoning_effort" in req:
        ctx["reasoning_effort"] = req["reasoning_effort"]
    return ctx


special = {k: getattr(tok, k) for k in ("bos_token", "eos_token", "unk_token", "pad_token")}
cases = []
for name, msgs in convs.items():
    msgs = srt(msgs)
    use_tools = tools if name == "tool_loop" else None
    for gen in (True, False):
        for effort in (None, "high"):
            for req in requests:
                req = dict(req, **({"reasoning_effort": effort} if effort else {}))
                ctx = mapped(req)
                expected = tok.apply_chat_template(msgs, tools=use_tools, chat_template=new, tokenize=False,
                                                   add_generation_prompt=gen, **ctx)
                shipped_render = tok.apply_chat_template(msgs, tools=use_tools, chat_template=shipped, tokenize=False,
                                                         add_generation_prompt=gen, **ctx)
                cases.append({"conv": name, "add_generation_prompt": gen, "request": req, "template_context": ctx,
                              "expected": expected, "equals_shipped": expected == shipped_render,
                              "expected_ids_tail": tok(expected, add_special_tokens=False)["input_ids"][-3:]})
fixture = {"template_sha256": hashlib.sha256(new.encode()).hexdigest(),
           "shipped_sha256": hashlib.sha256(shipped.encode()).hexdigest(), "special_tokens": special,
           "tools": tools, "conversations": {k: srt(v) for k, v in convs.items()}, "cases": cases}
(out / "renders.json").write_text(json.dumps(fixture, indent=1, ensure_ascii=False))
print(len(cases), "cases; equal to shipped:", sum(c["equals_shipped"] for c in cases))
