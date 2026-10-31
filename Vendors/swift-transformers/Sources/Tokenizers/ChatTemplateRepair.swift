// Copyright © 2026 Apple Inc.

import Foundation

/// Legacy template diagnostics. Selected bundle/override templates are rendered
/// unchanged; corrections belong to the template's source and bundle export.
public enum ChatTemplateRepair {
    /// Detects old Gemma spellings for explicit bundle diagnostics only.
    /// This does not validate the template or authorize a runtime rewrite.
    public static func needsRepair(_ template: String) -> Bool {
        template.contains("{%- elif not (ns_tr_out.flag and not has_content) -%}")
            || template.contains(
                "{%- if not (message['tool_responses'] and not message['content']) -%}")
    }

    /// Retained for source compatibility. The former blanket rewrite suppressed
    /// turn closure even before a new user and changed otherwise valid templates.
    @available(
        *, deprecated,
        message:
            "Update the model's template at its source; native templates are rendered unchanged."
    )
    public static func repaired(_ template: String) -> String {
        template
    }
}
