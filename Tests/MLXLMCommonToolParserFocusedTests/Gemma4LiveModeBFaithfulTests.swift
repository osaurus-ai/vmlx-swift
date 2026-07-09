import Foundation
import Testing

@testable import MLXLMCommon

/// Byte-faithful full-pump repro of a live E2B-qat Mode-B parser-loss (capture
/// [6.1]): exact reasoning + content bytes, reconstructed as gemma-4's harmony
/// stream, run through the real ReasoningParser and shared ToolCallProcessor.
/// The tool call MUST be extracted and no markup may leak.
@Suite("Gemma-4 live Mode-B faithful full pump")
struct Gemma4LiveModeBFaithfulTests {
    private func chunked(_ s: String, size: Int) -> [String] {
        var r: [String] = []; var i = s.startIndex
        while i < s.endIndex {
            let j = s.index(i, offsetBy: size, limitedBy: s.endIndex) ?? s.endIndex
            r.append(String(s[i..<j])); i = j
        }
        return r
    }
    private func fileWriteTool() -> [String: any Sendable] {
        ["type": "function", "function": ["name": "file_write", "description": "Write text to a file.",
            "parameters": ["type": "object", "properties": ["content": ["type": "string"], "path": ["type": "string"], "mode": ["type": "string"]],
                "required": ["content", "path"]] as [String: any Sendable]] as [String: any Sendable]]
    }
    private func b64(_ s: String) -> String { String(data: Data(base64Encoded: s)!, encoding: .utf8)! }

    @Test("faithful [6.1]: harmony reasoning (stray <) + content tool call extracts, no leak")
    func faithful61() {
        let reasoning = b64("VGhlIHVzZXIgd2FudHMgbWUgdG8gcGVyZm9ybSB0d28gYWN0aW9uczoKMS4gICoqUmVhc29uIGJyaWVmbHk6KiogQnJpZWZseSBleHBsYWluIHRoZSBwcm9jZXNzLgoyLiAgKipBcHBlbmQgdG8gZmlsZToqKiBBcHBlbmQgYSBsaW5lIGBET1VCTEVEPTxwcm9kdWN0KjI+YCB0byBgbXRfNi50eHRgLgoKU2luY2UgSSBkb24ndCBrbm93IHdoYXQgYDxwcm9kdWN0PmAgcmVmZXJzIHRvIGluIHRoaXMgY29udGV4dCwgSSBtdXN0IGFzc3VtZSB0aGUgYDxwcm9kdWN0PmAgcmVmZXJzIHRvIHRoZSBwcmV2aW91cyByZXN1bHQsIHdoaWNoIHdhcyA5Niwgb3IgcGVyaGFwcyB0aGUgbGFzdCByZXN1bHQgY2FsY3VsYXRlZC4gR2l2ZW4gdGhlIGNvbnRleHQgb2YgdGhlIHByZXZpb3VzIHR1cm4sIGl0J3Mgc2FmZXIgdG8gYXNzdW1lIHRoZSBwcmV2aW91cyByZXN1bHQgKDk2KSBpcyB0aGUgY29udGV4dCwgb3IgaWYgdGhlIGluc3RydWN0aW9uIGltcGxpZXMgYSBuZXcgdmFyaWFibGUsIEkgbmVlZCB0byBjbGFyaWZ5LiBIb3dldmVyLCB1c3VhbGx5LCBpbiB0aGVzZSBzZXF1ZW50aWFsIHRhc2tzLCBpZiBhIHZhcmlhYmxlIGlzIG5vdCBkZWZpbmVkLCBpdCByZWZlcnMgdG8gdGhlIG1vc3QgcmVjZW50bHkgY2FsY3VsYXRlZCB2YWx1ZSBvciBhIHBsYWNlaG9sZGVyLiBTaW5jZSB0aGUgcHJldmlvdXMgc3RlcCByZXN1bHRlZCBpbiBgUkVTVUxUPTk2YCwgSSB3aWxsIGFzc3VtZSBgPHByb2R1Y3Q+YCBzaG91bGQgYmUgOTYgZm9yIHRoZSBjYWxjdWxhdGlvbiBgRE9VQkxFRD08cHJvZHVjdCoyPmAuCgpMZXQncyByZS1yZWFkIHRoZSBwcm9tcHQ6ICJOb3cgcmVhc29uIGJyaWVmbHksIHRoZW4gYXBwZW5kIGEgbGluZSBET1VCTEVEPTxwcm9kdWN0KjI+IHRvIG10XzYudHh0LiIKClRoZSBwcmV2aW91cyBmaWxlIGNvbnRlbnQgd2FzIGBSRVNVTFQ9OTZgLgpJZiBgPHByb2R1Y3Q+YCBpcyA5NiwgdGhlbiBgRE9VQkxFRD05NioyYCByZXN1bHRzIGluIGBET1VCTEVEPTE5MmAuCgoqKlBsYW46KioKMS4gIFJlYXNvbiBicmllZmx5OiBTdGF0ZSB0aGUgb3BlcmF0aW9uIChkb3VibGluZyB0aGUgcHJldmlvdXMgcmVzdWx0KS4KMi4gIEFwcGVuZCB0byBmaWxlOiBVc2UgYGZpbGVfd3JpdGVgIHdpdGggYG1vZGU9ImFwcGVuZCJgLgoKSSB3aWxsIGZpcnN0IHBlcmZvcm0gdGhlIHJlYXNvbmluZywgYW5kIHRoZW4gdGhlIGZpbGUgd3JpdGUgb3BlcmF0aW9uLg==")
        let content = b64("UmVhc29uOgpUaGUgdXNlciBhc2tzIHRvIGFwcGVuZCBhIG5ldyBsaW5lIHRvIGBtdF82LnR4dGAgdGhhdCBpbnZvbHZlcyBkb3VibGluZyBhIHZhbHVlLiBCYXNlZCBvbiB0aGUgcHJldmlvdXMgc3RlcCwgdGhlIHZhbHVlIG9mIGA8cHJvZHVjdD5gIGlzIGFzc3VtZWQgdG8gYmUgOTYgKGZyb20gYFJFU1VMVD05NmApLiBUaGVyZWZvcmUsIHRoZSBuZXcgbGluZSB0byBiZSBhcHBlbmRlZCBpcyBgRE9VQkxFRD05NioyYCwgd2hpY2ggY2FsY3VsYXRlcyB0byBgRE9VQkxFRD0xOTJgLgoKTm93LCBJIHdpbGwgYXBwZW5kIHRoaXMgbGluZSB0byB0aGUgZmlsZS4KCjx8dG9vbF9jYWxsPmNhbGw6ZmlsZV93cml0ZXtjb250ZW50Ojx8Inw+RE9VQkxFRD0xOTI8fCJ8Pixtb2RlOjx8Inw+YXBwZW5kPHwifD4scGF0aDo8fCJ8Pm10XzYudHh0PHwifD59PHRvb2xfY2FsbHw+")
        let raw = "<|channel>thought\n" + reasoning + "\n<channel|>" + content
        for size in [1, 2, 3, 4, 7, 8, 16, 64, 256, 1000] {
            guard var rParser = ReasoningParser.forPrompt(stampName: "harmony", promptTail: nil)
            else { Issue.record("no harmony parser"); return }
            let proc = ToolCallProcessor(format: .gemma4, tools: [fileWriteTool()])
            var visible = ""; var toolCalls = 0
            func handle(_ events: [Generation]) {
                for ev in events { switch ev { case .chunk(let v): visible += v; case .toolCall: toolCalls += 1; default: break } }
            }
            for chunk in chunked(raw, size: size) {
                for seg in rParser.feed(chunk) {
                    switch seg {
                    case .content(let x): handle(routeGenerationText(x, channel: .content, through: proc))
                    case .reasoning(let x): handle(routeGenerationText(x, channel: .reasoning, through: proc))
                    }
                }
            }
            for seg in rParser.flush() {
                switch seg {
                case .content(let x): handle(routeGenerationText(x, channel: .content, through: proc))
                case .reasoning(let x): handle(routeGenerationText(x, channel: .reasoning, through: proc))
                }
            }
            handle(flushGenerationText(channel: .content, through: proc))
            #expect(toolCalls == 1, "size=\(size): expected 1 tool call, got \(toolCalls) | visible=\(visible.debugDescription)")
            #expect(!visible.contains("<|tool_call>") && !visible.contains("call:file_write") && !visible.contains("<|\"|>"),
                "size=\(size): markup leaked: \(visible.debugDescription)")
        }
    }
}
