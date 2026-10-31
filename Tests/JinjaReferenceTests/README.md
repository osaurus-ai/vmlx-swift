# Jinja reference controls

The tojson fixture follows Hugging Face transformers commit66f51e76efd8542232b8bdd40ade7d1edbaac46f, src/transformers/utils/chat_template_utils.py. Its filter delegates to Python json.dumps with ensure_ascii=False and preserves the Jinja object's order unless sort_keys is requested. The pinned primary file SHA256 is b8395f06ead2539c48e7d7f0fba5d085f9f30d6524ff2793546ed45b4a66b487.

67 JSON rendering/error cases and13 upper-coercion cases were generated with Python/Jinja2 3.1.6. Unordered host dictionaries are canonicalized before reference rendering, matching Value(any:). Explicit object literals separately test insertion order. This is CPU conformance coverage, not model-output or tool-name quality proof.
