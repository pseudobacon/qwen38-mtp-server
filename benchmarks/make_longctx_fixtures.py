#!/usr/bin/env python3
"""Build deterministic long-context fixtures from the real repo Swift source.

For each target token count (raw, no chat template), concatenate every Swift
source file from the server + engine repos (sorted by absolute path for
determinism), truncate at a token boundary to ~target, and write a fixture text
file. The exact prompt_tokens (with the Qwen chat template) are reported by
run_cell.sh at request time.

usage: make_longctx_fixtures.py
"""
import os
import sys
import bisect
from tokenizers import Tokenizer

SERVER = "/Users/cwong/ai/qwen38-mtp-server"
ENGINE = "/Users/cwong/ai/mlx-swift-lm"
OUTDIR = os.path.join(SERVER, "benchmarks", "prompts")

TARGETS = [
    ("longctx-8k.txt", 8192),
    ("longctx-32k.txt", 32768),
    ("longctx-96k.txt", 98304),
]


def gather_swift_files():
    roots = [
        os.path.join(SERVER, "Sources"),
        os.path.join(SERVER, "Tests"),
        os.path.join(ENGINE, "Libraries"),
        os.path.join(ENGINE, "Tests"),
    ]
    files = []
    for root in roots:
        for dirpath, _dirs, names in os.walk(root):
            for n in names:
                if n.endswith(".swift"):
                    files.append(os.path.join(dirpath, n))
    return sorted(files)


def build_corpus(files):
    parts = []
    total_lines = 0
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                body = fh.read()
        except OSError:
            continue
        parts.append("=====" + f + "=====\n" + body)
        total_lines += body.count("\n")
    corpus = "\n".join(parts)
    return corpus, files, total_lines


def main():
    tok = Tokenizer.from_file(os.path.join(SERVER, "weights", "tokenizer.json"))
    files = gather_swift_files()
    corpus, used, total_lines = build_corpus(files)
    total_tokens = len(tok.encode(corpus, add_special_tokens=False).ids)
    print(f"corpus: {len(files)} files, {total_lines} lines, {total_tokens} tokens")
    os.makedirs(OUTDIR, exist_ok=True)

    # char boundary -> token count, monotonic; binary search a prefix length.
    def tokens_of_prefix(n):
        return len(tok.encode(corpus[:n], add_special_tokens=False).ids)

    for name, target in TARGETS:
        if target > total_tokens:
            print(f"  {name}: target {target} > corpus {total_tokens}; using whole corpus")
            text = corpus
        else:
            lo, hi = 0, len(corpus)
            while lo < hi:
                mid = (lo + hi) // 2
                if tokens_of_prefix(mid) < target:
                    lo = mid + 1
                else:
                    hi = mid
            # lo is the smallest prefix length with >= target tokens; back off to <= target
            p = lo
            while p > 0 and tokens_of_prefix(p) > target:
                p -= 1
            text = corpus[:p]
            got = tokens_of_prefix(p)
        out_path = os.path.join(OUTDIR, name)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        ntok = tokens_of_prefix(len(text)) if len(text) < len(corpus) else total_tokens
        print(f"  {name}: {len(text)} chars, ~{ntok} raw tokens, sha256 prefix below")
        import hashlib
        print("    sha256:", hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
