"""Byte-level BPE tokenizer: text -> word pieces -> ids, and back. Pure Python (no packages).

The Tiny GPT Reader reads word pieces instead of single characters, so its memory holds about
4x more text: a passage plus a question. The browser has a TypeScript copy of the encoder
(portfolio: src/lib/tinygpt/bpe.ts) that must split text exactly the same way;
`python bpe.py vectors` writes test cases for it.

How it works
- Text is first cut into chunks (see `chunks`): words with their leading space, single digits,
  runs of punctuation, runs of whitespace. Merges never cross chunk borders.
- Each chunk becomes UTF-8 bytes (ids 0-255), so any character can be encoded: nothing is unknown.
- Training repeatedly merges the most frequent adjacent pair into a new id (256, 257, ...).
- Encoding a chunk repeats: find the adjacent pair with the earliest merge, merge every
  occurrence of it (left to right), until no pair has a merge.
- Special tokens (end of text, reader prompt markers) get the ids after the merges. They are
  never produced from text; the prompt builder inserts them.

    python bpe.py test                                    # self-test, no data needed
    python bpe.py train --input a.txt b.txt --vocab 8192 --out out/tokenizer.json
    python bpe.py vectors --tokenizer out/tokenizer.json --out out/bpe_vectors.json
"""

import argparse
import heapq
import json
import sys
import time
import unicodedata
from collections import Counter, defaultdict

FORMAT = "tinygpt-bpe-v1"
# End of text, then the reader's prompt markers: <|ctx|> passage <|q|> question <|a|> answer <|eos|>
SPECIALS = ["<|eos|>", "<|ctx|>", "<|q|>", "<|a|>", "<|noans|>"]

WS_EXTRA = "\t\n\r\x0b\x0c"


def char_class(c):
    """'ws', 'digit', 'letter' or 'other'. The TypeScript copy uses the same Unicode categories."""
    if c in WS_EXTRA:
        return "ws"
    cat = unicodedata.category(c)
    if cat == "Zs":
        return "ws"
    if cat == "Nd":
        return "digit"
    if cat[0] in "LM":  # letters, and the combining accents that belong to them
        return "letter"
    return "other"


def chunks(text):
    """Cuts text into the pieces merges may not cross.

    A chunk is an optional single leading space plus one of: a run of letters, one digit, or a
    run of other characters (punctuation, symbols). Whitespace runs are their own chunks, but
    the last space before a word is left for that word (" hello"), like GPT-2.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        start = i
        if text[i] == " " and i + 1 < n and char_class(text[i + 1]) != "ws":
            i += 1  # the space joins the word, digit or punctuation after it
        kind = char_class(text[i])
        if kind == "ws":
            j = i
            while j < n and char_class(text[j]) == "ws":
                j += 1
            if j < n and text[j - 1] == " " and j - 1 > i:
                j -= 1  # leave the last space for the next chunk
            out.append(text[i:j])
            i = j
            continue
        if kind == "digit":
            i += 1  # numbers are split into single digits
        else:
            while i < n and char_class(text[i]) == kind:
                i += 1
        out.append(text[start:i])
    return out


def _merge(ids, pair, new_id):
    """Every occurrence of `pair`, left to right, becomes `new_id`."""
    out, i, n = [], 0, len(ids)
    a, b = pair
    while i < n:
        if i + 1 < n and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges, specials=SPECIALS):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special = {name: 256 + len(self.merges) + i for i, name in enumerate(specials)}
        self.vocab_size = 256 + len(self.merges) + len(specials)
        self._bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self._bytes.append(self._bytes[a] + self._bytes[b])
        self._cache = {}

    def _encode_chunk(self, chunk):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            best = None
            for pair in zip(ids, ids[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best is None or rank < best):
                    best = rank
            if best is None:
                break
            ids = _merge(ids, self.merges[best], 256 + best)
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text):
        """Plain text to ids. Special tokens are never made from text."""
        ids = []
        for chunk in chunks(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def token_bytes(self, token_id):
        return self._bytes[token_id] if token_id < len(self._bytes) else b""

    def decode(self, ids):
        """Ids back to text; special tokens are left out."""
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace")

    def to_dict(self):
        return {
            "format": FORMAT,
            "vocab_size": self.vocab_size,
            "merges": [list(m) for m in self.merges],
            "special": self.special,
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data):
        if data.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} tokenizer")
        specials = sorted(data["special"], key=data["special"].get)
        return cls(data["merges"], specials)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def train(texts, vocab_size, specials=SPECIALS, log=print):
    """Learns merges from an iterable of texts until vocab_size (bytes + merges + specials)."""
    started = time.time()
    counts = Counter()
    for text in texts:
        counts.update(chunks(text))
    words = [list(chunk.encode("utf-8")) for chunk in counts]
    freqs = list(counts.values())
    log(f"{len(words):,} different chunks from {sum(freqs):,} in total ({time.time() - started:.0f} s)")

    pair_counts = Counter()
    where = defaultdict(set)  # pair -> indices of words that contain it
    for wi, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freqs[wi]
            where[pair].add(wi)
    # Max-heap by count; ties go to the smaller pair so training is repeatable. Stale entries
    # (counts that changed since they were pushed) are skipped when popped.
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges = []
    target = vocab_size - 256 - len(specials)
    while len(merges) < target and heap:
        neg, pair = heapq.heappop(heap)
        if pair_counts.get(pair, 0) != -neg:
            continue
        if -neg < 2:
            break  # nothing left that repeats
        new_id = 256 + len(merges)
        merges.append(pair)
        changed = set()
        for wi in list(where[pair]):
            word = words[wi]
            if len(word) < 2:
                continue
            old_pairs = Counter(zip(word, word[1:]))
            if pair not in old_pairs:
                continue
            new_word = _merge(word, pair, new_id)
            new_pairs = Counter(zip(new_word, new_word[1:]))
            f = freqs[wi]
            for p, k in old_pairs.items():
                pair_counts[p] -= k * f
                changed.add(p)
            for p, k in new_pairs.items():
                pair_counts[p] += k * f
                where[p].add(wi)
                changed.add(p)
            words[wi] = new_word
        del pair_counts[pair]
        where.pop(pair, None)
        for p in changed:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
            elif p in pair_counts:
                del pair_counts[p]
        if len(merges) % 1000 == 0:
            log(f"{len(merges):,} merges ({time.time() - started:.0f} s)")
    log(f"done: {len(merges):,} merges, vocabulary {256 + len(merges) + len(specials):,} "
        f"({time.time() - started:.0f} s)")
    return Tokenizer(merges, specials)


# Texts that exercise every rule: spaces, digits, accents, Filipino, emoji, tables, newlines.
VECTOR_TEXTS = [
    "",
    "Hello world",
    "  two leading spaces and trailing  ",
    "Line one\nLine two\n\n  indented line\r\nwindows line",
    "Tab\tseparated\tvalues",
    "Numbers: 347+28=375, 28-347=-319, 4,826,880 and 0.615.",
    "Prices like $1,376/month or ₱2,500.00 (VAT incl.)",
    "Contractions: don't, it's, we'll, I'M",
    "Ang mga bata ay naglalaro sa parke. Saan ka pupunta bukas?",
    "Café, naïve, résumé, São Paulo, Ñuñoa, piña",
    "Decomposed: Café and ñ",
    "Emoji 😀👍🏽 and symbols ©®™ → ≥ ½ ² Ⅳ",
    "中文字符和日本語のテキスト، والعربية",
    "File size | 10 MB per file",
    "<|eos|> is plain text here, not a special token",
    "Unusual   spacing with different spaces",
    "A" * 300,
    "The quick brown fox jumps over the lazy dog. " * 5,
]


def write_vectors(tok, path, extra_texts=()):
    cases = [{"text": t, "ids": tok.encode(t)} for t in list(VECTOR_TEXTS) + list(extra_texts)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"format": FORMAT, "cases": cases}, f, ensure_ascii=False)
    return len(cases)


def selftest():
    assert chunks("Hello world") == ["Hello", " world"]
    assert chunks("a  b") == ["a", " ", " b"]
    assert chunks("a\n b") == ["a", "\n", " b"]
    assert chunks(" 347+28") == [" 3", "4", "7", "+", "2", "8"]
    assert chunks("don't") == ["don", "'", "t"]
    assert chunks("end.  ") == ["end", ".", "  "]
    assert chunks("Café!") == ["Café", "!"]
    for text in VECTOR_TEXTS:
        assert "".join(chunks(text)) == text, f"chunks lost text: {text!r}"

    corpus = ["the cat sat on the mat. the cat ran.", "a dog sat on a log; 12 dogs ran!"] * 50
    tok = train(corpus, vocab_size=256 + 40 + len(SPECIALS), log=lambda *_: None)
    # 17 different chunks: training stops once every chunk is a single piece (23 merges).
    assert 0 < len(tok.merges) <= 40
    assert all(len(tok.encode(c)) == 1 for c in chunks(corpus[0])), "a training word wasn't merged fully"
    for text in VECTOR_TEXTS + corpus[:2]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"round trip failed: {text!r}"
        assert all(i < 256 + len(tok.merges) for i in ids), "text produced a special token"
    assert len(tok.encode("the cat sat")) < len("the cat sat".encode()), "merges had no effect"
    again = Tokenizer.from_dict(json.loads(json.dumps(tok.to_dict())))
    assert again.encode(corpus[0]) == tok.encode(corpus[0]), "save/load changed the encoding"
    assert again.special == tok.special
    print("bpe self-test passed")


def read_texts(paths):
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            yield f.read()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test")
    t = sub.add_parser("train")
    t.add_argument("--input", nargs="+", required=True)
    t.add_argument("--vocab", type=int, default=8192)
    t.add_argument("--out", default="out/tokenizer.json")
    v = sub.add_parser("vectors")
    v.add_argument("--tokenizer", default="out/tokenizer.json")
    v.add_argument("--out", default="out/bpe_vectors.json")
    v.add_argument("--sample", nargs="*", default=[], help="text files: their first 3 KB become extra cases")
    args = ap.parse_args()

    if args.cmd == "test":
        selftest()
    elif args.cmd == "train":
        tok = train(read_texts(args.input), args.vocab)
        tok.save(args.out)
        print(f"wrote {args.out}")
    elif args.cmd == "vectors":
        tok = Tokenizer.load(args.tokenizer)
        extra = [text[:3000] for text in read_texts(args.sample)]
        n = write_vectors(tok, args.out, extra)
        print(f"wrote {n} cases to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
