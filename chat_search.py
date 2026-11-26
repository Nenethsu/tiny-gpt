"""The portfolio chat's sentence picking, ported from its server (server/src/search.ts), so the
exams can test the agreement check: show the Reader's answer only when it lies inside the
sentence the search chose. Keep this in step with search.ts (stop words, stemmer, scoring).
Pure Python.
"""

import math
import re
import unicodedata

STOPWORDS = set((
    "a about above after again against all am an and any are as at be because been before being "
    "below between both but by can could did do does doing down during each few for from further "
    "had has have having he her here hers herself him himself his how i if in into is it its itself "
    "just me more most my myself no nor not now of off on once only or other our ours ourselves out "
    "over own same she should so some such than that the their theirs them themselves then there "
    "these they this those through to too under until up very was we were what when where which "
    "while who whom why will with would you your yours yourself yourselves tell please know "
    "explain describe give list show find say says said also may might must shall us let "
    "many much get gets got thing things anything something kind way "
    "ang ng sa na mga at ay ito iyan iyon si ni kay para ko mo niya namin natin nila ba po ka ako "
    "ikaw siya kami tayo sila din rin lang kung pero o ano sino saan kailan bakit paano ilan "
    "may mayroon wala hindi oo naman nga"
).split())

WORD = re.compile(r"[^\W_]+")  # letters and digits, like [\p{L}\p{N}]+
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[A-Z0-9À-Þ\"“'‘(\[])")


def plain(text):
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("M"))


def _has_vowel(w):
    return re.search(r"[aeiouy]", w) is not None


def stem(word):
    w = word
    if len(w) <= 2 or w.isdigit():
        return w
    if re.search(r"(?:ies|ied)$", w) and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif re.search(r"(?:x|z|ch|sh)es$", w):
        w = w[:-2]
    elif w.endswith("s") and not re.search(r"(?:ss|us|is)$", w) and len(w) > 3:
        w = w[:-1]
    if w.endswith("ing") and _has_vowel(w[:-3]) and len(w) > 4:
        w = w[:-3]
    elif w.endswith("ed") and not w.endswith("eed") and _has_vowel(w[:-2]) and len(w) > 3:
        w = w[:-2]
    if re.search(r"([b-df-hj-np-tv-z])\1$", w) and not re.search(r"(?:ll|ss|zz)$", w):
        w = w[:-1]
    if w.endswith("e") and not w.endswith("ee") and len(w) > 2:
        w = w[:-1]
    return w


def tokenize(text):
    return [stem(w) for w in WORD.findall(plain(text))
            if w not in STOPWORDS and (len(w) > 1 or any(c.isdigit() for c in w))]


def sentences(passage):
    out = []
    for s in SENTENCE_END.split(passage):
        s = s.strip()
        if s:
            words = s.split(" ")
            out += [" ".join(words[i:i + 60]) for i in range(0, len(words), 60)]
    return out


class Idf:
    """BM25 idf over a set of passages (the 'library'), as the server computes it."""

    def __init__(self, passages):
        self.n = len(passages)
        self.df = {}
        for p in passages:
            for t in set(tokenize(p)):
                self.df[t] = self.df.get(t, 0) + 1

    def __call__(self, t):
        d = self.df.get(t, 0)
        return math.log(1 + (self.n - d + 0.5) / (d + 0.5))


def is_label(sentence):
    return len(sentence.split(" ")) <= 6 and not re.search(r"[.!?:;]$", sentence) and " | " not in sentence


def best_sentence(passage, question, idf):
    """The sentence of `passage` the chat would show for `question` (server: bestSentence)."""
    query = tokenize(question)
    pairs = {f"{a} {b}" for a, b in zip(query, query[1:])}
    best, best_score = None, -1.0
    for s in sentences(passage):
        toks = tokenize(s)
        have = set(toks)
        score = sum(idf(t) for t in set(query) if t in have)
        score += 0.5 * sum(1 for a, b in zip(toks, toks[1:]) if f"{a} {b}" in pairs)
        if is_label(s):
            score /= 2
        if score > best_score:
            best, best_score = s, score
    return best or passage
