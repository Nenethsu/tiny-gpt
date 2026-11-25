"""Fine-tune the pretrained Tiny GPT Reader to answer questions from a passage (SQuAD 2.0).

Step 2 of 2 (step 1 is pretrain.py). Each example is one prompt and the answer to learn:

    <|eos|> <|ctx|> passage <|q|> question <|a|> answer <|eos|>
    <|eos|> <|ctx|> passage <|q|> question <|a|> <|noans|> <|eos|>     (the passage doesn't say)

The loss counts only the answer, so the model spends everything on answering. Answers are
copied from the passage, encoded exactly as they appear there (" Paris", with its space).
Passages are trimmed to about 150 words around the answer: the size of what the portfolio's
search hands the reader.

    python finetune.py --init /content/drive/MyDrive/tiny-gpt-reader --out /content/drive/MyDrive/tiny-gpt-reader/qa
    python finetune.py --preset smoke --init /content/reader-smoke --out /content/reader-qa-smoke
    python finetune.py --check-data --tokenizer out/tokenizer-test.json   # data only, no torch

The exam is the SQuAD 2.0 dev set (11,873 questions, never trained on): exact match and F1 as
SQuAD scores them, "no answer" accuracy, and how the portfolio's safety check would do (an
answer is shown only if it appears word for word in the passage).
"""

import argparse
import json
import os
import random
import re
import string
import time
import urllib.request
from collections import Counter

import bpe

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/{split}-v2.0.json"
MAX_WORDS = 150  # passage window, about the size of one search passage
MAX_ANSWER_TOKENS = 32

PRESETS = {
    "default": dict(iters=8000, batch_size=32, lr=3e-4, warmup=200, eval_every=1000,
                    eval_questions=500, save_every=250, max_train=None),
    "smoke": dict(iters=40, batch_size=8, lr=1e-3, warmup=5, eval_every=20,
                  eval_questions=20, save_every=20, max_train=2000),
}


# ---------------------------------------------------------------- data (pure Python)

def download(split, folder="data"):
    path = os.path.join(folder, f"squad-{split}-v2.json")
    if not os.path.exists(path):
        os.makedirs(folder, exist_ok=True)
        print(f"downloading SQuAD 2.0 {split}...")
        with urllib.request.urlopen(SQUAD_URL.format(split=split)) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)["data"]


SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def sentence_spans(text):
    spans, start = [], 0
    for m in SENTENCE_END.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def words(text):
    return len(text.split())


def window(context, rng, start=None, end=None):
    """A run of whole sentences of at most MAX_WORDS. With an answer (start, end), the run
    contains it, placed at a random spot. Returns (text, offset of the text in context)."""
    if words(context) <= MAX_WORDS:
        return context, 0
    spans = sentence_spans(context)
    if start is None:
        lo = hi = rng.randrange(len(spans))
    else:
        lo = next(i for i, (s, e) in enumerate(spans) if s <= start < e or start < s)
        hi = next((i for i, (s, e) in enumerate(spans) if end <= e), len(spans) - 1)
        hi = max(hi, lo)
    size = lambda a, b: words(context[spans[a][0]:spans[b][1]])  # noqa: E731
    if size(lo, hi) > MAX_WORDS:
        return None  # one huge sentence: skip the example
    while True:
        options = []
        if lo > 0 and size(lo - 1, hi) <= MAX_WORDS:
            options.append((lo - 1, hi))
        if hi < len(spans) - 1 and size(lo, hi + 1) <= MAX_WORDS:
            options.append((lo, hi + 1))
        if not options:
            break
        lo, hi = rng.choice(options)
    return context[spans[lo][0]:spans[hi][1]], spans[lo][0]


def prompt_ids(tok, passage, question):
    """The reader's prompt. The portfolio builds exactly the same one in the browser."""
    s = tok.special
    return ([s["<|eos|>"], s["<|ctx|>"]] + tok.encode(passage.strip())
            + [s["<|q|>"]] + tok.encode(question.strip()) + [s["<|a|>"]])


def answer_ids(tok, passage, start, text):
    """The answer as it appears in the passage: with its leading space, if it has one."""
    prefix = " " if start > 0 and passage[start - 1] == " " else ""
    return tok.encode(prefix + text) + [tok.special["<|eos|>"]]


def build_examples(articles, tok, rng, block_size, max_examples=None):
    """(prompt ids, answer ids, info) for every question; info keeps what the exam needs."""
    noans = [tok.special["<|noans|>"], tok.special["<|eos|>"]]
    out, skipped = [], 0
    for art in articles:
        for para in art["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if qa.get("is_impossible") or not qa["answers"]:
                    win = window(context, rng)
                    if win is None:
                        skipped += 1
                        continue
                    passage, _ = win
                    answers, ans = [], noans
                else:
                    first = qa["answers"][0]
                    start, text = first["answer_start"], first["text"]
                    if context[start:start + len(text)] != text:
                        skipped += 1
                        continue
                    win = window(context, rng, start, start + len(text))
                    if win is None:
                        skipped += 1
                        continue
                    passage, offset = win
                    local = start - offset
                    answers = [a["text"] for a in qa["answers"]
                               if 0 <= a["answer_start"] - offset and a["text"] in passage]
                    ans = answer_ids(tok, passage, local, text)
                    if len(ans) > MAX_ANSWER_TOKENS:
                        skipped += 1
                        continue
                prompt = prompt_ids(tok, passage, qa["question"])
                if len(prompt) + len(ans) > block_size:
                    skipped += 1
                    continue
                out.append((prompt, ans, {"passage": passage, "question": qa["question"], "answers": answers}))
                if max_examples and len(out) >= max_examples:
                    return out, skipped
    return out, skipped


# SQuAD's official answer comparison: lower case, no punctuation, no articles, single spaces.
def normalize(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def score(rows):
    """rows: (prediction or "" for no answer, gold answers ([] = no answer), passage)."""
    has, no = [r for r in rows if r[1]], [r for r in rows if not r[1]]

    def em_f1(pred, golds):
        if not golds:
            return float(pred == ""), float(pred == "")
        if pred == "":
            return 0.0, 0.0
        return (max(float(normalize(pred) == normalize(g)) for g in golds),
                max(f1(pred, g) for g in golds))

    results = [em_f1(p, g) for p, g, _ in rows]
    shown = [(p, g) for p, g, passage in rows if p and p in passage]
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else 0.0  # noqa: E731
    return {
        "count": len(rows),
        "exact_match": pct([r[0] for r in results]),
        "f1": pct([r[1] for r in results]),
        "has_answer_exact": pct([em_f1(p, g)[0] for p, g, _ in has]),
        "has_answer_f1": pct([em_f1(p, g)[1] for p, g, _ in has]),
        "no_answer_correct": pct([float(p == "") for p, _, _ in no]),
        # The portfolio shows an answer only if it's word for word in the passage.
        "shown": pct([float(bool(p and p in passage)) for p, _, passage in rows]),
        "shown_correct": pct([float(any(normalize(p) == normalize(x) for x in g)) for p, g in shown]),
    }


def check_data(tokenizer_path):
    """Builds examples from the dev set with a test tokenizer and checks them. No torch."""
    tok = bpe.Tokenizer.load(tokenizer_path)
    articles = download("dev")
    rng = random.Random(0)
    examples, skipped = build_examples(articles, tok, rng, block_size=512)
    s = tok.special
    has = [e for e in examples if e[1] != [s["<|noans|>"], s["<|eos|>"]]]
    lengths = sorted(len(p) + len(a) for p, a, _ in examples)
    # Can the answer be copied? Its tokens should appear in the passage tokens.
    copyable = 0
    for prompt, ans, _ in has:
        body = ans[:-1]
        if any(prompt[i:i + len(body)] == body for i in range(len(prompt) - len(body) + 1)):
            copyable += 1
    print(f"{len(examples):,} examples ({len(has):,} with an answer), {skipped} skipped")
    print(f"length in tokens: median {lengths[len(lengths) // 2]}, 95th percentile "
          f"{lengths[int(len(lengths) * 0.95)]}, longest {lengths[-1]}")
    print(f"answers whose tokens appear as-is in the passage: {100 * copyable / len(has):.1f}%")
    prompt, ans, info = has[0]
    print("example prompt:", repr(tok.decode(prompt)[:200]), "-> answer:", repr(tok.decode(ans)))
    # The scorer, on known cases.
    rows = [("Paris", ["Paris"], "in Paris."), ("the Paris", ["Paris"], "x"), ("", [], "x"),
            ("Lyon", ["Paris"], "Lyon"), ("", ["Paris"], "x")]
    sc = score(rows)
    assert sc["exact_match"] == 60.0 and sc["no_answer_correct"] == 100.0, sc
    assert sc["shown"] == 40.0 and sc["shown_correct"] == 50.0, sc
    assert abs(f1("Paris France", "Paris") - 2 / 3) < 1e-9
    print("scorer checks passed")


# ---------------------------------------------------------------- training (PyTorch)

def train(args, p):
    import math

    import torch

    from model import GPT, GPTConfig

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    init_ckpt = os.path.join(args.init, "ckpt.pt")
    if not os.path.exists(init_ckpt):
        raise SystemExit(f"{init_ckpt} not found: run pretrain.py first (or check --init).")
    tok = bpe.Tokenizer.load(os.path.join(args.init, "tokenizer.json"))
    os.makedirs(args.out, exist_ok=True)
    tok.save(os.path.join(args.out, "tokenizer.json"))

    base = torch.load(init_ckpt, map_location=device, weights_only=False)
    cfg = GPTConfig(**base["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(base["model"])
    print(f"loaded the pretrained model: {model.num_params():,} parameters, "
          f"pretrained for {base['step']:,} steps")

    started = time.time()
    rng = random.Random(1337)
    train_ex, skipped = build_examples(download("train"), tok, rng, cfg.block_size, p["max_train"])
    dev_ex, dev_skipped = build_examples(download("dev"), tok, random.Random(7), cfg.block_size)
    print(f"{len(train_ex):,} training questions ({skipped} skipped), {len(dev_ex):,} exam questions "
          f"({dev_skipped} skipped) ({time.time() - started:.0f} s)")
    eval_ex = random.Random(3).sample(dev_ex, min(p["eval_questions"], len(dev_ex)))

    eos, noans = tok.special["<|eos|>"], tok.special["<|noans|>"]

    def get_batch():
        batch = [train_ex[rng.randrange(len(train_ex))] for _ in range(p["batch_size"])]
        T = max(len(pr) + len(a) for pr, a, _ in batch) - 1
        x = torch.full((len(batch), T), eos, dtype=torch.long)
        y = torch.full((len(batch), T), -100, dtype=torch.long)  # -100: not part of the loss
        for r, (pr, a, _) in enumerate(batch):
            seq = pr + a
            x[r, : len(seq) - 1] = torch.tensor(seq[:-1])
            y[r, len(pr) - 1 : len(seq) - 1] = torch.tensor(a)  # predict only the answer
        return x.to(device), y.to(device)

    opt = model.configure_optimizer(weight_decay=0.1, lr=p["lr"], betas=(0.9, 0.95), device_type=device)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def autocast():
        return torch.autocast(device, dtype=torch.float16, enabled=use_amp)

    @torch.no_grad()
    def answer_all(examples):
        """Greedy answers, batched by prompt length so each row keeps its own positions."""
        model.eval()
        groups = {}
        for i, (pr, _, _) in enumerate(examples):
            groups.setdefault(len(pr), []).append(i)
        preds = [""] * len(examples)
        for n, idxs in groups.items():
            for k in range(0, len(idxs), 64):
                part = idxs[k : k + 64]
                ids = torch.tensor([examples[i][0] for i in part], device=device)
                for _ in range(MAX_ANSWER_TOKENS):
                    if ids.size(1) >= cfg.block_size:
                        break
                    with autocast():
                        logits, _ = model(ids)
                    nxt = logits[:, -1].argmax(-1, keepdim=True)
                    ids = torch.cat([ids, nxt], dim=1)
                    if (ids[:, n:] == eos).any(dim=1).all():
                        break
                for row, i in zip(ids[:, n:].tolist(), part):
                    out = row[: row.index(eos)] if eos in row else row
                    preds[i] = "" if (not out or out[0] == noans) else tok.decode(out).strip()
        model.train()
        return preds

    def exam(examples):
        preds = answer_all(examples)
        rows = [(pr, info["answers"], info["passage"]) for pr, (_, _, info) in zip(preds, examples)]
        return score(rows), preds

    ckpt_path = os.path.join(args.out, "ckpt.pt")
    log = {"base_steps": base["step"], "params": model.num_params(), "config": cfg.to_dict(),
           "dataset": "SQuAD 2.0 (Rajpurkar et al., 2018)", "train_questions": len(train_ex),
           "exam_questions": len(dev_ex), "iters": p["iters"], "batch_size": p["batch_size"],
           "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "loss": [], "exam": [], "train_seconds": 0}
    step = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        step, log = ck["step"], ck["log"]
        print(f"resuming at step {step:,} of {p['iters']:,}")

    def lr_at(s):
        if s < p["warmup"]:
            return p["lr"] * (s + 1) / p["warmup"]
        progress = (s - p["warmup"]) / max(1, p["iters"] - p["warmup"])
        return p["lr"] / 10 + 0.5 * (p["lr"] - p["lr"] / 10) * (1 + math.cos(math.pi * progress))

    session_start, elapsed_before, first = time.time(), log["train_seconds"], step

    def save():
        log["steps_done"] = step
        log["train_seconds"] = round(elapsed_before + time.time() - session_start)
        with open(os.path.join(args.out, "training.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "config": cfg.to_dict(), "step": step, "log": log}, ckpt_path + ".part")
        os.replace(ckpt_path + ".part", ckpt_path)

    recent = []
    while True:
        if step % p["eval_every"] == 0 or step == p["iters"]:
            sc, _ = exam(eval_ex)
            if not log["exam"] or log["exam"][-1]["step"] != step:
                log["exam"].append({"step": step, **sc})
            if recent:
                log["loss"].append({"step": step, "train": round(sum(recent) / len(recent), 4)})
                recent = []
            done = step - first
            left = (time.time() - session_start) / max(done, 1) * (p["iters"] - step)
            print(f"step {step:>5} | exact {sc['exact_match']}% | F1 {sc['f1']}% | no-answer "
                  f"{sc['no_answer_correct']}% | shown {sc['shown']}%, of those right {sc['shown_correct']}% | "
                  f"{(elapsed_before + time.time() - session_start) / 60:.1f} min"
                  + (f", ~{left / 60:.0f} min left" if done else ""))
            save()
        elif step % p["save_every"] == 0:
            save()
        if step >= p["iters"]:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch()
        with autocast():
            _, loss = model(x, y)
        recent.append(loss.item())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1

    print(f"final exam on all {len(dev_ex):,} questions it never trained on...")
    final, preds = exam(dev_ex)
    log["final"] = final
    examples = []
    for pred, (_, _, info) in list(zip(preds, dev_ex))[:: max(1, len(dev_ex) // 12)][:12]:
        examples.append({"question": info["question"], "expected": info["answers"][:1] or ["(no answer)"],
                         "got": pred or "(no answer)"})
    log["examples"] = examples
    save()
    print(json.dumps(final, indent=1))
    for e in examples[:6]:
        print(f"  Q: {e['question']}\n     expected {e['expected'][0]!r}, got {e['got']!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="default")
    ap.add_argument("--init", default="out-reader", help="the pretrain.py output folder")
    ap.add_argument("--out", default="out-reader/qa")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--device")
    ap.add_argument("--check-data", action="store_true", help="test the data side only (no torch)")
    ap.add_argument("--tokenizer", default="out/tokenizer-test.json", help="for --check-data")
    args = ap.parse_args()
    if args.check_data:
        check_data(args.tokenizer)
        return
    p = dict(PRESETS[args.preset])
    if args.iters:
        p["iters"] = args.iters
    train(args, p)


if __name__ == "__main__":
    main()
