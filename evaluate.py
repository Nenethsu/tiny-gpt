"""Exam the fine-tuned Reader with a confidence score, and pick the cut-off the portfolio uses.

For each exam question the Reader answers greedily, and its confidence is the geometric mean
of the probabilities of the tokens it chose (answer + end mark). The portfolio shows an answer
only if (1) it passes the safety check (word for word in the passage, or yes/no) and (2) its
confidence is at or above the cut-off; otherwise the chat falls back to the source sentence.
This script measures, for each cut-off, how often an answer is shown and how often shown
answers are right, and recommends the lowest cut-off whose shown answers are right at least
--target percent of the time.

    python evaluate.py --qa /content/drive/MyDrive/tiny-gpt-reader/qa              # GPU: every question
    python evaluate.py --qa /content/drive/MyDrive/tiny-gpt-reader/qa --limit 1500 # CPU: a sample

Writes calibration.json next to the checkpoint. Works on CPU (Colab's CPU runtime is fine):
answers are generated with a key/value cache, so each prompt is read once.
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
import finetune as ft
from model import GPT, GPTConfig

THRESHOLDS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


@torch.no_grad()
def generate(model, prompts, eos, max_new=ft.MAX_ANSWER_TOKENS):
    """Greedy answers for prompts of any lengths in one batch, with a key/value cache.
    Same math as GPT.forward. Returns, per prompt, the chosen tokens (up to and including the
    first eos) and their probabilities."""
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
    probs = [[] for _ in range(B)]
    done = [False] * B
    for _ in range(max_new):
        p = F.softmax(logits.float(), dim=-1)
        nxt = p.argmax(-1)
        pn = p[rows, nxt].tolist()
        for r, (t, q_) in enumerate(zip(nxt.tolist(), pn)):
            if not done[r]:
                chosen[r].append(t)
                probs[r].append(q_)
                done[r] = t == eos
        if all(done) or int(pos.max()) >= total:
            break
        x = model.wte(nxt)[:, None] + model.wpe(pos)[:, None]
        # Each row may attend to its own prompt and answer so far: positions 0..pos.
        mask = (torch.arange(total, device=dev)[None, :] <= pos[:, None])[:, None, None, :]
        for li, blk in enumerate(model.h):
            q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(C, dim=2)
            q, k, v = heads(q), heads(k), heads(v)
            K[li, rows, :, pos], V[li, rows, :, pos] = k[:, :, 0], v[:, :, 0]
            y = F.scaled_dot_product_attention(q, K[li], V[li], attn_mask=mask)
            x = x + blk.attn.c_proj(y.transpose(1, 2).reshape(B, 1, C))
            x = x + blk.mlp(blk.ln_2(x))
        logits = model.lm_head(model.ln_f(x))[:, 0]
        pos = pos + 1
    return chosen, probs


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


def answer(tok, tokens, probs):
    """(text or "" for no answer, confidence) from generated tokens."""
    eos, noans = tok.special["<|eos|>"], tok.special["<|noans|>"]
    conf = math.exp(sum(math.log(max(p, 1e-12)) for p in probs) / max(1, len(probs)))
    body = tokens[: tokens.index(eos)] if eos in tokens else tokens
    if not body or body[0] == noans:
        return "", conf
    return tok.decode(body).strip(), conf


def table(rows):
    """rows: (prediction, confidence, gold answers, passage). Per cut-off: shown and right."""
    out = []
    answerable = sum(1 for r in rows if r[2])
    for t in THRESHOLDS:
        shown = [(p, g) for p, c, g, passage in rows if p and c >= t and ft.shown(p, passage)]
        right = sum(1 for p, g in shown if any(ft.normalize(p) == ft.normalize(x) for x in g))
        out.append({
            "cutoff": t,
            "shown": round(100 * len(shown) / len(rows), 1),
            "right": round(100 * right / len(shown), 1) if shown else None,
            "answered_right": round(100 * right / answerable, 1) if answerable else 0.0,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="out-reader/qa", help="the finetune.py output folder")
    ap.add_argument("--limit", type=int, help="questions per exam (a random sample); default: all")
    ap.add_argument("--target", type=float, default=80.0, help="wanted %% right among shown answers")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device")
    args = ap.parse_args()

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

    # The fast generator must agree with the plain one.
    sample = exams["squad"][:6] + exams["coqa"][:6]
    fast, _ = generate(model, [p for p, _, _ in sample], eos)
    differ = [(got, want) for (prompt, _, _), got in zip(sample, fast)
              if got != (want := plain_greedy(model, prompt, eos))]
    # Rounding can flip a near tie now and then; more than one disagreement means a real bug.
    if len(differ) > 1:
        raise SystemExit(f"cached generation disagrees with plain greedy on {len(differ)} of 12: {differ[:2]}")
    print(f"cached generation matches plain greedy on {12 - len(differ)} of 12 questions")

    result = {"target_right": args.target, "exams": {}}
    all_rows = []
    for name, examples in exams.items():
        started = time.time()
        order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))  # similar lengths together
        rows = [None] * len(examples)
        for k in range(0, len(order), args.batch):
            part = order[k : k + args.batch]
            toks, probs = generate(model, [examples[i][0] for i in part], eos)
            for i, t, pr in zip(part, toks, probs):
                pred, conf = answer(tok, t, pr)
                info = examples[i][2]
                rows[i] = (pred, conf, info["answers"], info["passage"])
            if (k // args.batch) % 20 == 0:
                done = k + len(part)
                rate = done / (time.time() - started)
                print(f"  {name}: {done:,} of {len(examples):,} ({rate:.1f} questions/s, "
                      f"~{(len(examples) - done) / rate / 60:.0f} min left)")
        sc = ft.score([(p, g, passage) for p, _, g, passage in rows])
        result["exams"][name] = {"score": sc, "cutoffs": table(rows)}
        all_rows += rows
        print(f"{name.upper()} ({len(rows):,}): exact {sc['exact_match']}%, F1 {sc['f1']}%, "
              f"shown {sc['shown']}% ({sc['shown_correct']}% right) in {(time.time() - started) / 60:.1f} min")

    combined = table(all_rows)
    result["combined"] = combined
    pick = next((r for r in combined if r["right"] is not None and r["right"] >= args.target), None)
    result["recommended_cutoff"] = pick["cutoff"] if pick else None
    print("\ncut-off | shown | right among shown | right among answerable questions")
    for r in combined:
        print(f"  {r['cutoff']:<5} | {r['shown']:>5}% | {str(r['right']) + '%':>6} | {r['answered_right']}%")
    if pick:
        print(f"\nrecommended cut-off: {pick['cutoff']} (answers shown {pick['shown']}% of the time, "
              f"{pick['right']}% of those right)")
    else:
        print(f"\nno cut-off reaches {args.target}% right; the best is shown above")
    with open(os.path.join(args.qa, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {args.qa}/calibration.json")


if __name__ == "__main__":
    main()
