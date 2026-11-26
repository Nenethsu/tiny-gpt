"""Export a fine-tuned Reader for the browser (portfolio: public/models/tiny-gpt-reader/).

    python export_reader.py --root /content/drive/MyDrive/tiny-gpt-reader --out /content/reader-web
    python export_reader.py --root ... --model qa      # the older Reader that writes its answer

--model span (default) reads root/span/ (finetune_span.py: the Reader that points at the answer);
--model qa reads root/qa/ (finetune.py). Writes:
    model.json      config and tensor table; "tokenizer" names the word-piece file, "head" says
                    which answer head the model has ("span": start/end scores, or "lm": words)
    model.bin       the weights (int8 matrices, float32 biases), same format as Tiny GPT
    tokenizer.json  the byte-level BPE tokenizer (bpe.py; the portfolio's bpe.ts reads it)
    check.json      what PyTorch computes from the exported weights for a real prompt, for the
                    portfolio's engine check (span: start/end scores at every position + the answer)
    reader.json     what the page shows and how to answer: model size, pretraining and fine-tuning
                    stats, exam scores, and the confidence cut-off
Runs on CPU (Colab's CPU runtime is fine).
"""

import argparse
import json
import math
import os

import torch

import bpe
import finetune as ft
import finetune_span as fs
import weights_format as wf
from model import GPT, GPTConfig

CHECK_PASSAGE = ("The Normans were the people who in the 10th and 11th centuries gave their name to "
                 "Normandy, a region in France. They were descended from Norse raiders and pirates.")
CHECK_HISTORY = [("Who gave their name to Normandy?", "The Normans")]
CHECK_QUESTION = "Where is Normandy?"


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def best_answer(tok, ids, lay, start, end):
    """The answer the span head picks, exactly as finetune_span.py's exam does (and the browser
    must): log-softmax over allowed starts / ends, best passage pair at most MAX_SPAN tokens
    long, or the yes / no / no-answer slots. Returns (text, confidence)."""
    p0, p1 = lay["passage"]
    starts = list(range(p0, p1)) + [lay["yes"][0], lay["no"][0], lay["noans"]]
    ends = list(range(p0, p1)) + [lay["yes"][1], lay["no"][1], lay["noans"]]

    def log_softmax(scores, allowed):
        m = max(scores[i] for i in allowed)
        z = m + math.log(sum(math.exp(scores[i] - m) for i in allowed))
        return {i: scores[i] - z for i in allowed}

    ls, le = log_softmax(start, starts), log_softmax(end, ends)
    choices = [(ls[s] + le[e], s, e) for s in range(p0, p1) for e in range(s, min(p1, s + fs.MAX_SPAN))]
    for a, b in (lay["yes"], lay["no"], (lay["noans"], lay["noans"])):
        choices.append((ls[a] + le[b], a, b))
    score, s, e = max(choices)
    return fs.span_text(tok, ids, lay, s, e), math.exp(score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out-reader", help="pretrain.py output (span/ or qa/ inside it)")
    ap.add_argument("--model", choices=("span", "qa"), default="span")
    ap.add_argument("--out", default="out/reader-web")
    args = ap.parse_args()
    src = os.path.join(args.root, args.model)

    tok = bpe.Tokenizer.load(os.path.join(src, "tokenizer.json"))
    ck = torch.load(os.path.join(src, "ckpt.pt"), map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    head = None
    if args.model == "span":
        weights = ck["model"]
        model.load_state_dict({k[len("gpt."):]: v for k, v in weights.items() if k.startswith("gpt.")})
        head = {"span.weight": weights["span.weight"].float(), "span.bias": weights["span.bias"].float()}
    else:
        model.load_state_dict(ck["model"])
    model.eval()
    state = model.state_dict()
    params = model.num_params() + sum(t.numel() for t in (head or {}).values())

    print("writing weights (int8 rows; about a minute)...")
    writer = wf.Writer()
    for name, quantize in wf.tensor_names(cfg.n_layer):
        t = state[name].float().contiguous()
        writer.add(name, list(t.shape), t.view(-1).tolist(), quantize)
    for name, t in (head or {}).items():
        writer.add(name, list(t.shape), t.contiguous().view(-1).tolist(), False)  # tiny: keep float32
    config = {k: getattr(cfg, k) for k in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")}
    manifest = writer.save(args.out, config, "", tok.special["<|eos|>"],
                           extra={"params": params, "tokenizer": "tokenizer.json", "special": tok.special,
                                  "head": "span" if head else "lm"})
    tok.save(os.path.join(args.out, "tokenizer.json"))
    print(f"model.bin: {manifest['bytes'] / 1e6:.2f} MB ({params:,} parameters)")

    # PyTorch on the exported (int8) weights: the browser engine must give the same numbers.
    _, tensors = wf.read(args.out)
    with torch.no_grad():
        for name, (shape, flat) in tensors.items():
            if name in state:
                state[name].copy_(torch.tensor(flat, dtype=torch.float32).view(shape))
        check = {"passage": CHECK_PASSAGE, "history": CHECK_HISTORY, "question": CHECK_QUESTION}
        if head:
            ids, lay = fs.span_prompt(tok, CHECK_PASSAGE, CHECK_QUESTION, CHECK_HISTORY)
            idx = torch.tensor([ids])
            x = model.wte(idx) + model.wpe(torch.arange(len(ids)))
            for block in model.h:
                x = block(x)
            W = torch.tensor(tensors["span.weight"][1]).view(2, cfg.n_embd)
            b = torch.tensor(tensors["span.bias"][1])
            scores = model.ln_f(x)[0] @ W.T + b
            start, end = scores[:, 0].tolist(), scores[:, 1].tolist()
            text, conf = best_answer(tok, ids, lay, start, end)
            check |= {"kind": "span", "ids": ids, "layout": lay, "start": start, "end": end,
                      "answer": {"text": text, "confidence": conf}}
            print(f"check prompt: {len(ids)} tokens; int8 model answers {text!r} (confidence {conf:.3f})")
        else:
            ids = ft.prompt_ids(tok, CHECK_PASSAGE, CHECK_QUESTION, CHECK_HISTORY)
            logits, _ = model(torch.tensor([ids]))
            positions = sorted(set(list(range(0, len(ids), max(1, len(ids) // 8))) + [len(ids) - 1]))
            check |= {"kind": "lm", "ids": ids, "positions": positions,
                      "logits": [logits[0, p].tolist() for p in positions]}
    with open(os.path.join(args.out, "check.json"), "w", encoding="utf-8") as f:
        json.dump(check, f)

    pre = read_json(os.path.join(args.root, "training.json")) or {}
    fine = read_json(os.path.join(src, "training.json")) or {}
    cal = read_json(os.path.join(src, "calibration.json"))
    reader = {
        "kind": "span" if head else "lm",
        "params": params,
        "config": config,
        "download_bytes": manifest["bytes"],
        "tokenizer": {"vocab_size": tok.vocab_size, "chars_per_token": pre.get("chars_per_token")},
        "pretraining": {k: pre.get(k) for k in ("dataset", "articles", "characters", "train_tokens",
                                                "steps_done", "train_seconds", "device")}
                       | {"final_loss": (pre.get("loss") or [None])[-1]},
        "finetuning": {k: fine.get(k) for k in ("datasets", "train_questions", "exam_questions",
                                                "steps_done", "train_seconds", "device")},
        "exams": fine.get("final"),
        "examples": fine.get("examples"),
        # How the browser should answer: the cut-off and how often shown answers were right.
        "decoding": cal.get("recommended") if cal else None,
        "max_span": fs.MAX_SPAN,
        "history_turns": ft.HISTORY_TURNS,
    }
    with open(os.path.join(args.out, "reader.json"), "w", encoding="utf-8") as f:
        json.dump(reader, f, indent=1)
    if not cal:
        print("note: no calibration.json yet, so reader.json has no cut-off")
    print(f"wrote {args.out}/. Copy these files to the portfolio's public/models/tiny-gpt-reader/")


if __name__ == "__main__":
    main()
