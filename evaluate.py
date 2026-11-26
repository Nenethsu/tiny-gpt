"""Exam the fine-tuned Reader with a confidence score, and pick how the portfolio should use it.

Two ways of answering are compared:
- free: plain greedy decoding (the model may write anything);
- copy: each token must continue a phrase that appears in the passage, or be " yes" / " no"
  or <|noans|> (forced copying). Every answer then passes the safety check by design.

Confidence is the geometric mean of the chosen tokens' probabilities, measured two ways: "raw"
(the model's own probability) and "allowed" (renormalized over the tokens copying allows).
The portfolio shows the Reader's answer only if it passes the safety check (word for word in the
passage, or yes/no) and its confidence is at or above a cut-off; otherwise the chat shows the
source sentence. For each way and cut-off this script reports how often an answer is shown, how
often shown answers are exactly right, and how often they are useful (F1 >= 0.5 with a right
answer). It recommends the combination that shows the most answers while shown answers are
exactly right at least --target percent of the time.

Agreement check: the chat shows a search sentence anyway, so an answer can also be required to
lie inside the sentence the chat's search picks (chat_search.py, ported from the server). Two
methods agreeing is often more reliable than either alone. Yes/no answers have no span, so they
are counted on their own.

    python evaluate.py --qa /content/drive/MyDrive/tiny-gpt-reader/qa --limit 1500   # CPU, ~10 min
    python evaluate.py --qa /content/drive/MyDrive/tiny-gpt-reader/qa --modes free copy
    python evaluate.py --qa /content/drive/MyDrive/tiny-gpt-reader/qa --from-predictions  # seconds

Writes calibration.json and predictions.json (every answer, so later checks need no model run)
next to the checkpoint.
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

import bpe
import chat_search
import finetune as ft
from model import GPT, GPTConfig

THRESHOLDS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
MODES = ("free", "copy")
CONFIDENCES = ("raw", "allowed")


def passage_tokens(tok, prompt):
    """The passage part of a reader prompt: between <|ctx|> and the first <|q|>."""
    start = prompt.index(tok.special["<|ctx|>"]) + 1
    return prompt[start : prompt.index(tok.special["<|q|>"], start)]


class CopyRule:
    """Which tokens may come next when the answer must be copied from the passage (or be
    " yes", " no", or <|noans|>). Tracks every place the answer so far could have come from."""

    def __init__(self, tok, passage):
        self.eos, self.noans = tok.special["<|eos|>"], tok.special["<|noans|>"]
        self.sources = [passage, tok.encode(" yes"), tok.encode(" no")]
        self.matches = [(s, i) for s, seq in enumerate(self.sources) for i in range(len(seq))]
        self.length = 0  # tokens copied so far
        self.closed = False  # after <|noans|> only the end mark may follow

    def allowed(self):
        if self.closed:
            return {self.eos}
        nxt = {self.sources[s][i + self.length] for s, i in self.matches
               if i + self.length < len(self.sources[s])}
        if self.length == 0:
            nxt.add(self.noans)
        else:
            nxt.add(self.eos)
        return nxt

    def push(self, token):
        if token == self.noans:
            self.closed = True
            return
        self.matches = [(s, i) for s, i in self.matches
                        if i + self.length < len(self.sources[s]) and self.sources[s][i + self.length] == token]
        self.length += 1


@torch.no_grad()
def generate(model, prompts, eos, rules=None, max_new=ft.MAX_ANSWER_TOKENS):
    """Greedy answers for prompts of any lengths in one batch, with a key/value cache (same math
    as GPT.forward). With `rules` (one CopyRule per prompt), tokens a rule doesn't allow are
    masked out. Returns per prompt: chosen tokens (up to and including the first eos), their raw
    probabilities, and their probabilities among the allowed tokens."""
    cfg = model.cfg
    B, H, C = len(prompts), cfg.n_head, cfg.n_embd
    hd = C // H
    dev = model.wte.weight.device
    lengths = torch.tensor([len(p) for p in prompts], device=dev)
    T = int(lengths.max())
    total = min(cfg.block_size, T + max_new)
    tokens = torch.full((B, T), eos, dtype=torch.long, device=dev)
    for r, p in enumerate(prompts):
        tokens[r, : len(p)] = torch.tensor(p, device=dev)
    K = torch.zeros(cfg.n_layer, B, H, total, hd, device=dev)
    V = torch.zeros_like(K)

    def heads(t):  # (B, T, C) -> (B, H, T, hd)
        return t.view(t.size(0), t.size(1), H, hd).transpose(1, 2)

    # The prompts, all at once. Padding sits after each prompt, so causal attention never sees it.
    x = model.wte(tokens) + model.wpe(torch.arange(T, device=dev))
    for li, blk in enumerate(model.h):
        q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(C, dim=2)
        q, k, v = heads(q), heads(k), heads(v)
        K[li, :, :, :T], V[li, :, :, :T] = k, v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + blk.attn.c_proj(y.transpose(1, 2).reshape(B, T, C))
        x = x + blk.mlp(blk.ln_2(x))
    logits = model.lm_head(model.ln_f(x))[torch.arange(B, device=dev), lengths - 1]

    rows = torch.arange(B, device=dev)
    pos = lengths.clone()  # where each row's next token goes
    chosen = [[] for _ in range(B)]
    raw = [[] for _ in range(B)]
    within = [[] for _ in range(B)]
    done = [False] * B
    for _ in range(max_new):
        p = F.softmax(logits.float(), dim=-1)
        if rules is None:
            q_ = p
        else:
            mask = torch.zeros_like(p, dtype=torch.bool)
            for r, rule in enumerate(rules):
                mask[r, list(rule.allowed())] = True
            q_ = p * mask
            q_ = q_ / q_.sum(-1, keepdim=True).clamp_min(1e-12)
        nxt = q_.argmax(-1)
        p_raw, p_in = p[rows, nxt].tolist(), q_[rows, nxt].tolist()
        for r, t in enumerate(nxt.tolist()):
            if not done[r]:
                chosen[r].append(t)
                raw[r].append(p_raw[r])
                within[r].append(p_in[r])
                done[r] = t == eos
                if rules is not None and not done[r]:
                    rules[r].push(t)
        if all(done) or int(pos.max()) >= total:
            break
        x = model.wte(nxt)[:, None] + model.wpe(pos)[:, None]
        # Each row may attend to its own prompt and answer so far: positions 0..pos.
        att = (torch.arange(total, device=dev)[None, :] <= pos[:, None])[:, None, None, :]
        for li, blk in enumerate(model.h):
            q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(C, dim=2)
            q, k, v = heads(q), heads(k), heads(v)
            K[li, rows, :, pos], V[li, rows, :, pos] = k[:, :, 0], v[:, :, 0]
            y = F.scaled_dot_product_attention(q, K[li], V[li], attn_mask=att)
            x = x + blk.attn.c_proj(y.transpose(1, 2).reshape(B, 1, C))
            x = x + blk.mlp(blk.ln_2(x))
        logits = model.lm_head(model.ln_f(x))[:, 0]
        pos = pos + 1
    return chosen, raw, within


@torch.no_grad()
def plain_greedy(model, prompt, eos, max_new=ft.MAX_ANSWER_TOKENS):
    """The slow, obvious way (whole sequence every step), to check `generate` against."""
    ids = list(prompt)
    out = []
    for _ in range(max_new):
        logits, _ = model(torch.tensor([ids], device=model.wte.weight.device))
        t = int(logits[0, -1].argmax())
        out.append(t)
        ids.append(t)
        if t == eos or len(ids) >= model.cfg.block_size:
            break
    return out


def geo_mean(probs):
    return math.exp(sum(math.log(max(p, 1e-12)) for p in probs) / max(1, len(probs)))


def answer(tok, tokens):
    """The answer text ("" for no answer) from generated tokens."""
    eos, noans = tok.special["<|eos|>"], tok.special["<|noans|>"]
    body = tokens[: tokens.index(eos)] if eos in tokens else tokens
    if not body or body[0] == noans:
        return ""
    return tok.decode(body).strip()


def table(rows, confidence, allow=lambda r: True):
    """rows: dicts with pred, raw, allowed, golds, passage, agree. Per cut-off: shown, right,
    useful. `allow` adds a condition (e.g. the agreement check)."""
    out = []
    answerable = sum(1 for r in rows if r["golds"])
    for t in THRESHOLDS:
        shown = [r for r in rows if r["pred"] and r[confidence] >= t and ft.shown(r["pred"], r["passage"])
                 and allow(r)]
        right = sum(1 for r in shown if any(ft.normalize(r["pred"]) == ft.normalize(g) for g in r["golds"]))
        useful = sum(1 for r in shown if r["golds"] and max(ft.f1(r["pred"], g) for g in r["golds"]) >= 0.5)
        out.append({
            "cutoff": t,
            "shown": round(100 * len(shown) / len(rows), 1),
            "right": round(100 * right / len(shown), 1) if shown else None,
            "useful": round(100 * useful / len(shown), 1) if shown else None,
            "answered_right": round(100 * right / answerable, 1) if answerable else 0.0,
        })
    return out


def print_table(title, rows):
    print(f"\n{title}\ncut-off | shown | right | useful | right among answerable questions")
    for r in rows:
        right = "-" if r["right"] is None else f"{r['right']}%"
        useful = "-" if r["useful"] is None else f"{r['useful']}%"
        print(f"  {r['cutoff']:<5} | {r['shown']:>5}% | {right:>6} | {useful:>6} | {r['answered_right']}%")


FILTERS = {
    "cut-off only": lambda r: True,
    "agreement + cut-off": lambda r: r["agree"],
    "yes/no only": lambda r: ft.is_yes_no(r["pred"]),
}


def add_agreement(rows):
    """Marks each row: does its (non yes/no) answer lie inside the chat search's best sentence?
    The exam passages play the library, for idf."""
    idf = chat_search.Idf(list({r["passage"] for r in rows}))
    for r in rows:
        if r["pred"] and not ft.is_yes_no(r["pred"]):
            r["sentence"] = chat_search.best_sentence(r["passage"], r["question"], idf)
            r["agree"] = r["pred"] in r["sentence"]
        else:
            r["agree"] = False
    return rows


def run_model(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    tok = bpe.Tokenizer.load(os.path.join(args.qa, "tokenizer.json"))
    ck = torch.load(os.path.join(args.qa, "ckpt.pt"), map_location=device, weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    eos = tok.special["<|eos|>"]
    print(f"Reader: {model.num_params():,} parameters, fine-tuned {ck['step']:,} steps, on {device}")

    exams = {
        "squad": ft.build_squad(ft.squad("dev"), tok, random.Random(7), cfg.block_size)[0],
        "coqa": ft.build_coqa(ft.coqa("dev"), tok, random.Random(7), cfg.block_size, exam=True)[0],
    }
    if args.limit:
        exams = {k: random.Random(11).sample(v, min(args.limit, len(v))) for k, v in exams.items()}

    # The fast generator must agree with the plain one (free decoding).
    sample = exams["squad"][:6] + exams["coqa"][:6]
    fast, _, _ = generate(model, [p for p, _, _ in sample], eos)
    differ = [(got, want) for (prompt, _, _), got in zip(sample, fast)
              if got != (want := plain_greedy(model, prompt, eos))]
    # Rounding can flip a near tie now and then; more than one disagreement means a real bug.
    if len(differ) > 1:
        raise SystemExit(f"cached generation disagrees with plain greedy on {len(differ)} of 12: {differ[:2]}")
    print(f"cached generation matches plain greedy on {12 - len(differ)} of 12 questions")

    predictions = {}
    for mode in args.modes:
        predictions[mode] = []
        for name, examples in exams.items():
            started = time.time()
            order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))  # similar lengths together
            rows = [None] * len(examples)
            for k in range(0, len(order), args.batch):
                part = order[k : k + args.batch]
                prompts = [examples[i][0] for i in part]
                rules = [CopyRule(tok, passage_tokens(tok, p)) for p in prompts] if mode == "copy" else None
                toks, raw, within = generate(model, prompts, eos, rules)
                for i, t, pr, pw in zip(part, toks, raw, within):
                    info = examples[i][2]
                    rows[i] = {"exam": name, "question": info["question"], "history": info.get("history", []),
                               "pred": answer(tok, t), "raw": geo_mean(pr), "allowed": geo_mean(pw),
                               "golds": info["answers"], "passage": info["passage"]}
                if (k // args.batch) % 20 == 0:
                    done = k + len(part)
                    rate = done / (time.time() - started)
                    print(f"  {mode} / {name}: {done:,} of {len(examples):,} ({rate:.1f} questions/s, "
                          f"~{(len(examples) - done) / rate / 60:.0f} min left)")
            predictions[mode] += rows
            print(f"  {mode} / {name}: done in {(time.time() - started) / 60:.1f} min")
    with open(os.path.join(args.qa, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="out-reader/qa", help="the finetune.py output folder")
    ap.add_argument("--limit", type=int, help="questions per exam (a random sample); default: all")
    ap.add_argument("--target", type=float, default=80.0, help="wanted %% exactly right among shown answers")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=["free"])
    ap.add_argument("--from-predictions", action="store_true",
                    help="reuse predictions.json from an earlier run instead of running the model")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device")
    args = ap.parse_args()

    if args.from_predictions:
        with open(os.path.join(args.qa, "predictions.json"), encoding="utf-8") as f:
            predictions = json.load(f)
        print("reusing predictions.json: " + ", ".join(f"{m} ({len(r):,} answers)" for m, r in predictions.items()))
    else:
        predictions = run_model(args)

    result = {"target_right": args.target, "modes": {}}
    candidates = []
    for mode, rows in predictions.items():
        add_agreement(rows)
        result["modes"][mode] = {"exams": {}, "tables": {}}
        for name in ("squad", "coqa"):
            part = [r for r in rows if r["exam"] == name]
            sc = ft.score([(r["pred"], r["golds"], r["passage"]) for r in part])
            result["modes"][mode]["exams"][name] = sc
            print(f"{mode.upper()} / {name.upper()} ({len(part):,}): exact {sc['exact_match']}%, F1 {sc['f1']}%, "
                  f"no-answer {sc['no_answer_correct']}%, shown {sc['shown']}% ({sc['shown_correct']}% right)")
        spans = [r for r in rows if r["pred"] and not ft.is_yes_no(r["pred"]) and ft.shown(r["pred"], r["passage"])]
        agree = [r for r in spans if r["agree"]]
        print(f"  answers inside the search's sentence: {len(agree):,} of {len(spans):,} span answers "
              f"({100 * len(agree) / max(1, len(spans)):.1f}%)")
        confidences = CONFIDENCES if mode == "copy" else ("raw",)
        for c in confidences:
            for fname, allow in FILTERS.items():
                rows_t = table(rows, c, allow)
                result["modes"][mode]["tables"][f"{c} / {fname}"] = rows_t
                print_table(f"{mode} decoding, {c} confidence, {fname} (both exams)", rows_t)
                if fname == "yes/no only":
                    continue  # reported for a closer look, not a way to answer on its own
                for row in rows_t:
                    if row["right"] is not None and row["right"] >= args.target:
                        candidates.append({"decoding": mode, "confidence": c, "filter": fname, **row})
                        break

    best = max(candidates, key=lambda r: r["shown"]) if candidates else None
    result["recommended"] = best
    if best:
        print(f"\nrecommended: {best['decoding']} decoding, {best['confidence']} confidence, {best['filter']}, "
              f"cut-off {best['cutoff']}: answers shown {best['shown']}% of the time, {best['right']}% of those "
              f"exactly right, {best['useful']}% useful")
    else:
        print(f"\nno combination reaches {args.target}% exactly right; the tables above show the best")
    with open(os.path.join(args.qa, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {args.qa}/calibration.json")


if __name__ == "__main__":
    main()
