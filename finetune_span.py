"""Fine-tune the pretrained Reader to point at the answer in the passage (span extraction).

The other fine-tune (finetune.py) teaches the Reader to *write* its answer token by token. This
one teaches it to *point*: for every passage token it scores how likely the answer starts there
and how likely it ends there, and the answer is the best start/end pair. That is the standard
way to do this task; the model can't wander off the passage, and comparing every possible
phrase gives a more honest confidence. It reuses the same pretraining (pretrain.py).

The question comes first, so every passage token can see it (attention only looks back). The
end of the prompt holds three extra choices the model can point at instead:

    <|eos|> [<|q|> earlier question <|a|> earlier answer]... <|q|> question <|ctx|> passage <|a|> " yes" " no" <|noans|>

Confidence = p(start) * p(end), each a softmax over the places the answer may start / end.

    python finetune_span.py --init /content/drive/MyDrive/tiny-gpt-reader --out /content/drive/MyDrive/tiny-gpt-reader/span
    python finetune_span.py --preset smoke --init /content/reader-smoke --out /content/reader-span-smoke
    python finetune_span.py --check-data --tokenizer out/tokenizer-test.json   # data only, no torch

Same exams as finetune.py (SQuAD 2.0 dev, CoQA dev with history). Writes predictions.json and
calibration.json in the same format as evaluate.py, so `evaluate.py --qa <out> --from-predictions`
can add the agreement tables.
"""

import argparse
import bisect
import json
import os
import random
import time

import bpe
import finetune as ft

MAX_SPAN = 30  # answer length limit, in tokens

PRESETS = {
    "default": dict(iters=10000, batch_size=32, lr=2e-4, head_lr=1e-3, warmup=300, eval_every=1000,
                    eval_questions=300, save_every=250, max_train=None),
    "smoke": dict(iters=40, batch_size=8, lr=1e-3, head_lr=1e-3, warmup=5, eval_every=20,
                  eval_questions=20, save_every=20, max_train=2000),
}


# ---------------------------------------------------------------- data (pure Python)

def span_prompt(tok, passage, question, history=()):
    """The pointing prompt and where its parts are: (ids, layout). The portfolio builds the same."""
    s = tok.special
    ids = [s["<|eos|>"]]
    for q, a in list(history)[-ft.HISTORY_TURNS:]:
        ids += [s["<|q|>"]] + tok.encode(q.strip()) + [s["<|a|>"]] + tok.encode(" " + a.strip())
    ids += [s["<|q|>"]] + tok.encode(question.strip()) + [s["<|ctx|>"]]
    p0 = len(ids)
    pids = tok.encode(passage)
    ids += pids
    p1 = len(ids)
    ids.append(s["<|a|>"])
    yes, no = tok.encode(" yes"), tok.encode(" no")
    yes_span = (len(ids), len(ids) + len(yes) - 1)
    ids += yes
    no_span = (len(ids), len(ids) + len(no) - 1)
    ids += no
    noans = len(ids)
    ids.append(s["<|noans|>"])
    return ids, {"passage": (p0, p1), "yes": yes_span, "no": no_span, "noans": noans}


def token_starts(tok, pids):
    """Byte offset where each passage token starts."""
    starts, at = [], 0
    for t in pids:
        starts.append(at)
        at += len(tok.token_bytes(t))
    return starts


def char_span_to_tokens(tok, passage, pids, start, end):
    """Passage characters [start, end) -> the (first, last) passage tokens that cover them."""
    starts = token_starts(tok, pids)
    b0 = len(passage[:start].encode("utf-8"))
    b1 = len(passage[:end].encode("utf-8"))
    return bisect.bisect_right(starts, b0) - 1, bisect.bisect_right(starts, b1 - 1) - 1


def target_for(tok, passage, question, history, kind, start=None, end=None):
    """(ids, layout, (start position, end position)) for one example."""
    ids, lay = span_prompt(tok, passage, question, history)
    p0 = lay["passage"][0]
    if kind == "noans":
        target = (lay["noans"], lay["noans"])
    elif kind in ("yes", "no"):
        target = lay[kind]
    else:
        pids = ids[p0 : lay["passage"][1]]
        a, b = char_span_to_tokens(tok, passage, pids, start, end)
        target = (p0 + a, p0 + b)
    return ids, lay, target


def build_squad(articles, tok, rng, block_size, max_examples=None):
    out, skipped = [], 0
    for art in articles:
        for para in art["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if qa.get("is_impossible") or not qa["answers"]:
                    win = ft.window(context, rng)
                    if win is None:
                        skipped += 1
                        continue
                    passage, golds, made = win[0], [], None
                    made = target_for(tok, passage, qa["question"], (), "noans")
                else:
                    first = qa["answers"][0]
                    start, text = first["answer_start"], first["text"]
                    win = ft.window(context, rng, start, start + len(text)) if context[start:start + len(text)] == text else None
                    if win is None:
                        skipped += 1
                        continue
                    passage, offset = win
                    golds = [a["text"] for a in qa["answers"] if a["text"] in passage]
                    made = target_for(tok, passage, qa["question"], (), "span", start - offset, start - offset + len(text))
                ids, lay, target = made
                if len(ids) > block_size or target[1] - target[0] >= MAX_SPAN:
                    skipped += 1
                    continue
                out.append((ids, lay, target, {"passage": passage, "question": qa["question"],
                                               "answers": golds, "history": []}))
                if max_examples and len(out) >= max_examples:
                    return out, skipped
    return out, skipped


def build_coqa(stories, tok, rng, block_size, max_examples=None, exam=False):
    """Like finetune.build_coqa: reworded answers are skipped as targets (kept in the history);
    the exam keeps every turn and grades against all the annotators' answers."""
    out, skipped = [], 0
    for story in stories:
        context = story["story"]
        extra = story.get("additional_answers", {})
        history = []
        for q, a in zip(story["questions"], story["answers"]):
            question, said = q["input_text"], a["input_text"].strip()
            clean = said.rstrip(".").strip()
            golds = [clean] + [x[a["turn_id"] - 1]["input_text"].strip().rstrip(".")
                               for x in extra.values() if len(x) >= a["turn_id"]]
            span_start = max(a["span_start"], 0)
            span_end = a["span_end"] if a["span_end"] >= 0 else span_start
            win = ft.window(context, rng, span_start, max(span_end, span_start + 1))
            made = None
            if win is not None:
                passage, offset = win
                if ft.normalize(clean) == "unknown":
                    golds = []
                    made = target_for(tok, passage, question, history, "noans")
                elif ft.is_yes_no(clean):
                    made = target_for(tok, passage, question, history, ft.normalize(clean))
                else:
                    hit = ft.coqa_target(passage, clean, span_start - offset)
                    if hit:
                        made = target_for(tok, passage, question, history, "span", hit[0], hit[0] + len(hit[1]))
                if made is None and exam:
                    made = target_for(tok, passage, question, history, "noans")  # graded against golds anyway
            if made is not None:
                ids, lay, target = made
                if len(ids) <= block_size and target[1] - target[0] < MAX_SPAN:
                    out.append((ids, lay, target, {"passage": passage, "question": question,
                                                   "answers": golds, "history": list(history[-ft.HISTORY_TURNS:])}))
                else:
                    skipped += 1
            else:
                skipped += 1
            history.append((question, said))
            if max_examples and len(out) >= max_examples:
                return out, skipped
    return out, skipped


def span_text(tok, ids, lay, s, e):
    """What pointing at (s, e) means: "" for no answer, "yes", "no", or the passage phrase."""
    if s == lay["noans"]:
        return ""
    if (s, e) == tuple(lay["yes"]):
        return "yes"
    if (s, e) == tuple(lay["no"]):
        return "no"
    return tok.decode(ids[s : e + 1]).strip()


def check_data(tokenizer_path):
    tok = bpe.Tokenizer.load(tokenizer_path)
    for name, build, data in (("SQuAD dev", build_squad, ft.squad("dev")), ("CoQA dev", build_coqa, ft.coqa("dev"))):
        examples, skipped = build(data, tok, random.Random(0), 512)
        spans = [e for e in examples if e[2][0] < e[1]["passage"][1]]
        exact = sum(1 for ids, lay, (s, t), info in spans
                    if any(ft.normalize(span_text(tok, ids, lay, s, t)) == ft.normalize(g) for g in info["answers"]))
        lengths = sorted(len(e[0]) for e in examples)
        kinds = {"noans": 0, "yes": 0, "no": 0, "span": 0}
        for ids, lay, (s, t), _ in examples:
            kinds["noans" if s == lay["noans"] else "yes" if (s, t) == lay["yes"] else "no" if (s, t) == lay["no"] else "span"] += 1
        print(f"{name}: {len(examples):,} examples, {skipped} skipped, {kinds}; tokens median "
              f"{lengths[len(lengths) // 2]}, longest {lengths[-1]}; target span reads back as a right answer: "
              f"{100 * exact / max(1, len(spans)):.1f}%")
    ids, lay, (s, t), info = next(e for e in build_coqa(ft.coqa("dev"), tok, random.Random(0), 512)[0] if e[3]["history"])
    print("example:", info["history"], "|", repr(info["question"]), "-> points at", repr(span_text(tok, ids, lay, s, t)))
    print("prompt ends with:", repr(tok.decode(ids[-6:])), "| layout:", lay)


# ---------------------------------------------------------------- training (PyTorch)

def train(args, p):
    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import evaluate as ev
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

    class SpanReader(nn.Module):
        """The pretrained GPT without its word output, plus a start/end score per position."""

        def __init__(self):
            super().__init__()
            self.gpt = GPT(cfg)
            self.span = nn.Linear(cfg.n_embd, 2)
            nn.init.normal_(self.span.weight, std=0.02)
            nn.init.zeros_(self.span.bias)

        def forward(self, idx):
            g = self.gpt
            x = g.wte(idx) + g.wpe(torch.arange(idx.size(1), device=idx.device))
            for block in g.h:
                x = block(x)
            out = self.span(g.ln_f(x))
            return out[..., 0], out[..., 1]  # start, end scores: (B, T) each

    model = SpanReader().to(device)
    model.gpt.load_state_dict(base["model"])
    print(f"loaded the pretrained model ({base['step']:,} steps) + a span head: "
          f"{sum(p_.numel() for p_ in model.parameters()):,} parameters")

    started = time.time()
    rng = random.Random(1337)
    half = p["max_train"] // 2 if p["max_train"] else None
    squad_train, s1 = build_squad(ft.squad("train"), tok, rng, cfg.block_size, half)
    coqa_train, s2 = build_coqa(ft.coqa("train"), tok, rng, cfg.block_size, half)
    train_ex = squad_train + coqa_train
    exams = {
        "squad": build_squad(ft.squad("dev"), tok, random.Random(7), cfg.block_size)[0],
        "coqa": build_coqa(ft.coqa("dev"), tok, random.Random(7), cfg.block_size, exam=True)[0],
    }
    print(f"training: {len(squad_train):,} SQuAD questions + {len(coqa_train):,} CoQA turns ({s1 + s2} skipped); "
          f"exams: {len(exams['squad']):,} SQuAD, {len(exams['coqa']):,} CoQA ({time.time() - started:.0f} s)")
    quick = {k: random.Random(3).sample(v, min(p["eval_questions"], len(v))) for k, v in exams.items()}
    eos = tok.special["<|eos|>"]
    NEG = -1e4  # "not allowed" (fits in float16)

    def allowed_masks(batch, T):
        """Where answers may start / end: the passage, the yes/no spans, the no-answer token."""
        starts = torch.zeros(len(batch), T, dtype=torch.bool)
        ends = torch.zeros(len(batch), T, dtype=torch.bool)
        for r, (_, lay, _, _) in enumerate(batch):
            p0, p1 = lay["passage"]
            starts[r, p0:p1] = ends[r, p0:p1] = True
            for a, b in (lay["yes"], lay["no"]):
                starts[r, a] = ends[r, b] = True
            starts[r, lay["noans"]] = ends[r, lay["noans"]] = True
        return starts, ends

    def tensors(batch):
        T = max(len(e[0]) for e in batch)
        x = torch.full((len(batch), T), eos, dtype=torch.long)
        for r, (ids, _, _, _) in enumerate(batch):
            x[r, : len(ids)] = torch.tensor(ids)
        sm, em = allowed_masks(batch, T)
        return x.to(device), sm.to(device), em.to(device)

    # The span head learns from scratch, so it gets a higher learning rate than the pretrained body.
    body = [q for q in model.gpt.parameters()]
    decay = [q for q in body if q.dim() >= 2]
    no_decay = [q for q in body if q.dim() < 2]
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.1, "lr": p["lr"], "base": p["lr"]},
        {"params": no_decay, "weight_decay": 0.0, "lr": p["lr"], "base": p["lr"]},
        {"params": list(model.span.parameters()), "weight_decay": 0.0, "lr": p["head_lr"], "base": p["head_lr"]},
    ], betas=(0.9, 0.95), fused=device == "cuda")
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def autocast():
        return torch.autocast(device, dtype=torch.float16, enabled=use_amp)

    @torch.no_grad()
    def predict(examples, batch=48):
        """Best answer and its confidence for every example (one forward pass each)."""
        model.eval()
        order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
        preds = [None] * len(examples)
        for k in range(0, len(order), batch):
            part = [examples[i] for i in order[k : k + batch]]
            x, sm, em = tensors(part)
            with autocast():
                st, en = model(x)
            st = F.log_softmax(st.float().masked_fill(~sm, NEG), dim=-1)
            en = F.log_softmax(en.float().masked_fill(~em, NEG), dim=-1)
            for r, (ids, lay, _, _) in enumerate(part):
                p0, p1 = lay["passage"]
                S, E = st[r, p0:p1], en[r, p0:p1]
                n = p1 - p0
                pair = S[:, None] + E[None, :]
                i = torch.arange(n, device=pair.device)
                band = (i[None, :] >= i[:, None]) & (i[None, :] - i[:, None] < MAX_SPAN)
                pair = pair.masked_fill(~band, -1e9)
                best = int(pair.argmax())
                choices = [(float(pair.view(-1)[best]), p0 + best // n, p0 + best % n)]
                for a, b in (lay["yes"], lay["no"], (lay["noans"], lay["noans"])):
                    choices.append((float(st[r, a] + en[r, b]), a, b))
                score, s, e = max(choices)
                preds[order[k + r]] = (span_text(tok, ids, lay, s, e), math.exp(score))
        model.train()
        return preds

    def exam(examples):
        preds = predict(examples)
        sc = ft.score([(pr, info["answers"], info["passage"]) for (pr, _), (_, _, _, info) in zip(preds, examples)])
        return sc, preds

    ckpt_path = os.path.join(args.out, "ckpt.pt")
    log = {"kind": "span", "base_steps": base["step"], "params": sum(q.numel() for q in model.parameters()),
           "config": cfg.to_dict(), "datasets": ["SQuAD 2.0 (Rajpurkar et al., 2018)", "CoQA (Reddy et al., 2019)"],
           "train_questions": {"squad": len(squad_train), "coqa": len(coqa_train)},
           "exam_questions": {k: len(v) for k, v in exams.items()}, "iters": p["iters"],
           "batch_size": p["batch_size"], "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "loss": [], "exam": [], "train_seconds": 0}
    step = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        step, log = ck["step"], ck["log"]
        print(f"resuming at step {step:,} of {p['iters']:,}")

    def lr_factor(s):
        if s < p["warmup"]:
            return (s + 1) / p["warmup"]
        progress = (s - p["warmup"]) / max(1, p["iters"] - p["warmup"])
        return 0.1 + 0.45 * (1 + math.cos(math.pi * progress))

    session_start, elapsed_before, first = time.time(), log["train_seconds"], step

    def save():
        log["steps_done"] = step
        log["train_seconds"] = round(elapsed_before + time.time() - session_start)
        with open(os.path.join(args.out, "training.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "config": cfg.to_dict(), "step": step, "log": log}, ckpt_path + ".part")
        os.replace(ckpt_path + ".part", ckpt_path)

    def summary(sc):
        return (f"exact {sc['exact_match']}%, F1 {sc['f1']}%, no-answer {sc['no_answer_correct']}%, "
                f"shown {sc['shown']}% ({sc['shown_correct']}% right)")

    recent = []
    while True:
        if step % p["eval_every"] == 0 or step == p["iters"]:
            results = {k: exam(v)[0] for k, v in quick.items()}
            if not log["exam"] or log["exam"][-1]["step"] != step:
                log["exam"].append({"step": step, **results})
            if recent:
                log["loss"].append({"step": step, "train": round(sum(recent) / len(recent), 4)})
                recent = []
            done = step - first
            left = (time.time() - session_start) / max(done, 1) * (p["iters"] - step)
            print(f"step {step:>5} | {(elapsed_before + time.time() - session_start) / 60:.1f} min"
                  + (f", ~{left / 60:.0f} min left" if done else "")
                  + f"\n    SQuAD: {summary(results['squad'])}\n    CoQA:  {summary(results['coqa'])}")
            save()
        elif step % p["save_every"] == 0:
            save()
        if step >= p["iters"]:
            break
        for g in opt.param_groups:
            g["lr"] = g["base"] * lr_factor(step)
        batch = [train_ex[rng.randrange(len(train_ex))] for _ in range(p["batch_size"])]
        x, sm, em = tensors(batch)
        ys = torch.tensor([b[2][0] for b in batch], device=device)
        ye = torch.tensor([b[2][1] for b in batch], device=device)
        with autocast():
            st, en = model(x)
        st = st.float().masked_fill(~sm, NEG)
        en = en.float().masked_fill(~em, NEG)
        loss = (F.cross_entropy(st, ys) + F.cross_entropy(en, ye)) / 2
        recent.append(loss.item())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1

    print("final exams on every question it never trained on...")
    log["final"], log["examples"], predictions = {}, {}, []
    for name, examples in exams.items():
        final, preds = exam(examples)
        log["final"][name] = final
        for (pred, conf), (_, _, _, info) in zip(preds, examples):
            predictions.append({"exam": name, "question": info["question"], "history": info["history"],
                                "pred": pred, "raw": conf, "allowed": conf, "golds": info["answers"],
                                "passage": info["passage"]})
        picks = list(zip(preds, examples))[:: max(1, len(examples) // 10)][:10]
        log["examples"][name] = [{"history": info["history"], "question": info["question"],
                                  "expected": info["answers"][:1] or ["(no answer)"], "got": pred or "(no answer)",
                                  "confidence": round(conf, 3)} for (pred, conf), (_, _, _, info) in picks]
        print(f"{name.upper()} ({final['count']:,}): {summary(final)}")
    save()
    with open(os.path.join(args.out, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump({"span": predictions}, f)

    table = ev.table(predictions, "raw")
    ev.print_table("span reader, confidence cut-off (both exams)", table)
    pick = next((r for r in table if r["right"] is not None and r["right"] >= args.target), None)
    with open(os.path.join(args.out, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"target_right": args.target, "modes": {"span": {"tables": {"raw / cut-off only": table}}},
                   "recommended": ({"decoding": "span", "confidence": "raw", "filter": "cut-off only", **pick}
                                   if pick else None)}, f, indent=1)
    if pick:
        print(f"\nrecommended cut-off {pick['cutoff']}: answers shown {pick['shown']}% of the time, "
              f"{pick['right']}% of those exactly right, {pick['useful']}% useful")
    else:
        print(f"\nno cut-off reaches {args.target}% exactly right; the table above shows the best")
    for e in log["examples"]["coqa"][:5]:
        before = " / ".join(f"{q} -> {a}" for q, a in e["history"])
        print(f"  [{before}]\n  Q: {e['question']}  expected {e['expected'][0]!r}, got {e['got']!r} ({e['confidence']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="default")
    ap.add_argument("--init", default="out-reader", help="the pretrain.py output folder")
    ap.add_argument("--out", default="out-reader/span")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--target", type=float, default=80.0)
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
