"""Resolve explicit fixture inputs without depending on a developer's home path."""

import argparse
import hashlib
import json
from pathlib import Path


def paths(fixture_name):
    parser = argparse.ArgumentParser(description=f"Regenerate {fixture_name} numerics")
    parser.add_argument("--reference", type=Path, required=True,
                        help="Matching JANG v26_audio.py or v26_vision.py reference")
    parser.add_argument("--output", type=Path, required=True,
                        help="Separate output directory for comparison")
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parents[2] / "Tests/MLXLMTests/Fixtures/MiMoV26"
    metadata = json.loads((fixtures / f"{fixture_name}-reference.json").read_text())
    source = args.reference.resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != metadata["reference_sha256"]:
        parser.error(f"reference SHA256 mismatch: {digest}; expected {metadata['reference_sha256']}")
    output = args.output.resolve()
    if output == fixtures:
        parser.error("generate into a separate directory and compare before replacing fixtures")
    output.mkdir(parents=True, exist_ok=True)
    return output, source
