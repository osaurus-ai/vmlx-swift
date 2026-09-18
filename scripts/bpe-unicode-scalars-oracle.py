#!/usr/bin/env python3
r"""Expected tokens and ids for Tests/MLXLMTests/BPETokenizerUnicodeScalarTests.swift.

Each case is a tiny BPE model, a vocabulary and a merge list with no normalizer or pre-tokenizer,
under which one of three grapheme-cluster faults changes the output: seeding the merges from
clusters, joining the pieces on " " and splitting them again, or comparing merge keys canonically.
This builds each case's tokenizer JSON, encodes its text with Hugging Face tokenizers (the library
transformers' fast tokenizers run), and prints the tokens and ids the test expects. It is development
tooling and not part of any build.

Strings that NFC or NFD would change are written with escapes, so an editor that normalizes this
file cannot change a case silently. Devanagari and Thai stay literal: normalization leaves them as
they are.

The printed values were generated with tokenizers 0.22.2 on Python 3.12:
    <venv>/bin/pip install "tokenizers==0.22.2"
    <venv>/bin/python scripts/bpe-unicode-scalars-oracle.py
"""

import json

from tokenizers import Tokenizer

E_ACUTE = "\N{LATIN SMALL LETTER E WITH ACUTE}"  # U+00E9, precomposed: NFC
ACUTE = "\N{COMBINING ACUTE ACCENT}"  # U+0301: "e" + ACUTE is the same letter in NFD

# name: (text, vocabulary in id order, merges in rank order). Keep in step with the Swift test.
CASES = {
    "devanagari": (
        "किसानों",
        ["क", "ि", "स", "ा", "न", "ो", "ं", "कि", "किस", "ान", "ानो", "ानों"],
        [("क", "ि"), ("कि", "स"), ("ा", "न"), ("ान", "ो"), ("ानो", "ं")],
    ),
    "crlf": ("\r\n", ["\r", "\n"], []),
    "skin-toned emoji": ("\U0001F44D\U0001F3FD", ["\U0001F44D", "\U0001F3FD"], []),
    "nfd accent": ("te" + ACUTE, ["t", "e", ACUTE, "te", "e" + ACUTE], [("t", "e"), ("e", ACUTE)]),
    "thai": ("สวัส", ["ส", "ว", "ั"], []),
    "byte fallback": ("\U0001F44D\U0001F3FD", ["\U0001F44D", "<0xF0>", "<0x9F>", "<0x8F>", "<0xBD>"], []),
    "merge keys": (
        E_ACUTE + "t" + E_ACUTE,
        [E_ACUTE, "t", E_ACUTE + "t", "t" + E_ACUTE, "e" + ACUTE, "e" + ACUTE + "t"],
        [(E_ACUTE, "t"), ("t", E_ACUTE), ("e" + ACUTE, "t")],
    ),
    "space": (" ", [" "], []),
    "inner space": ("a b", ["a", " ", "b", "a ", "a b"], [("a", " "), ("a ", "b")]),
}


def scalars(text):
    return " ".join(f"U+{ord(c):04X}" for c in text)


def shown(token):
    """A token as the Swift test's failure message shows it: printable ASCII as is, else code points."""
    return token if all("!" <= c <= "~" for c in token) else scalars(token)


for name, (text, vocab, merges) in CASES.items():
    model = {
        "type": "BPE",
        "vocab": {token: i for i, token in enumerate(vocab)},
        "merges": [list(merge) for merge in merges],
        "byte_fallback": True,
    }
    encoding = Tokenizer.from_str(json.dumps({"version": "1.0", "model": model})).encode(text)
    print(f"{name}: {scalars(text)}")
    print(f"  tokens {json.dumps(encoding.tokens, ensure_ascii=False)}")
    print(f"         {' | '.join(shown(token) for token in encoding.tokens)}")
    print(f"  ids    {encoding.ids}")
