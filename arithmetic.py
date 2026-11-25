"""Addition and subtraction practice for Tiny GPT, written the way the browser asks:

    question "347+28="  ->  answer "375"        question "28-347="  ->  answer "-319"

Pure Python (no torch) so it can be tested anywhere:  python arithmetic.py
"""

import random
import re

MAX_DIGITS = 3


def operand(rng, max_digits):
    """Picks the digit count first, so 7 is as common as 734 (visitors type small sums too)."""
    digits = rng.randint(1, max_digits)
    low = 0 if digits == 1 else 10 ** (digits - 1)
    return rng.randint(low, 10**digits - 1)


def problem(rng, max_digits=MAX_DIGITS):
    a, b = operand(rng, max_digits), operand(rng, max_digits)
    op = rng.choice("+-")
    return f"{a}{op}{b}=", str(a + b if op == "+" else a - b)


def solve(question):
    """The right answer for a question like "347+28=" (used by the tests and the page check)."""
    m = re.fullmatch(r"(\d+)([+-])(\d+)=", question)
    if not m:
        raise ValueError(f"not a problem: {question!r}")
    a, op, b = int(m[1]), m[2], int(m[3])
    return str(a + b if op == "+" else a - b)


def full_size(question, max_digits=MAX_DIGITS):
    a, b = re.split(r"[+-]", question[:-1])
    return max(len(a), len(b)) == max_digits


def make_sets(n_train, n_val, n_test, max_digits=MAX_DIGITS, seed=0):
    """Returns (train docs, val docs, test pairs). A doc is question + answer, e.g. "347+28=375".

    Test problems never appear in training or validation, so the score shows real skill, not memory.
    Only problems with at least one full-size number are held out; every small sum like 3+4 is
    still practised, because there are too few of them to spare.
    """
    rng = random.Random(seed)
    test = {}
    while len(test) < n_test:
        q, a = problem(rng, max_digits)
        if full_size(q, max_digits):
            test[q] = a

    def draw(n):
        docs = []
        while len(docs) < n:
            q, a = problem(rng, max_digits)
            if q not in test:
                docs.append(q + a)
        return docs

    return draw(n_train), draw(n_val), sorted(test.items())


def score(rows):
    """rows: (question, expected, model's answer). Percent correct overall and per operation."""

    def pct(subset):
        return round(100 * sum(e == g for _, e, g in subset) / len(subset), 2) if subset else None

    return {
        "overall": pct(rows),
        "add": pct([r for r in rows if "+" in r[0]]),
        "sub": pct([r for r in rows if "+" not in r[0]]),
        "count": len(rows),
    }


if __name__ == "__main__":
    train, val, test = make_sets(20000, 500, 300, seed=1)
    held = {q for q, _ in test}
    assert not any(d[: d.index("=") + 1] in held for d in train + val), "test problem leaked into training"
    assert all(solve(d[: d.index("=") + 1]) == d[d.index("=") + 1 :] for d in train[:2000])
    assert all(full_size(q) for q in held)
    assert solve("28-347=") == "-319" and solve("999+999=") == "1998" and solve("0-0=") == "0"
    sizes = [max(map(len, re.split(r"[+-]", d[: d.index("=")]))) for d in train]
    print("ok:", len(train), "train docs, e.g.", train[:5])
    print("largest operand has 1/2/3 digits:", [sizes.count(k) for k in (1, 2, 3)])
    print("score check:", score([("1+1=", "2", "2"), ("5-9=", "-4", "4")]))
