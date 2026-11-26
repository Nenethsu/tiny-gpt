"""Export the fine-tuned Reader for the browser (portfolio: public/models/tiny-gpt-reader/).

    python export_reader.py --root /content/drive/MyDrive/tiny-gpt-reader --out /content/reader-web

Reads root/qa/ckpt.pt (fine-tuned), root/qa/tokenizer.json, and the logs. Writes:
    model.json      config, tensor table; "tokenizer" names the word-piece file (no char vocab)
    model.bin       the weights (int8 matrices, float32 biases), same format as Tiny GPT
    tokenizer.json  the byte-level BPE tokenizer (bpe.py; the portfolio's bpe.ts reads it)
    check.json      logits PyTorch computes from the exported weights at a few positions of a
                    real reader prompt, for the portfolio's engine check
    reader.json     what the page shows: model size, pretraining and fine-tuning stats, exam
                    scores, and how to decode and gate answers (evaluate.py)
Runs on CPU (Colab's CPU runtime is fine).
"""

import argparse
import json
import os

import torch

import bpe
import finetune as ft
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out-reader", help="pretrain.py output (qa/ inside it)")
    ap.add_argument("--out", default="out/reader-web")
    args = ap.parse_args()
    qa = os.path.join(args.root, "qa")

    tok = bpe.Tokenizer.load(os.path.join(qa, "tokenizer.json"))
    ck = torch.load(os.path.join(qa, "ckpt.pt"), map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    state = model.state_dict()

    print("writing weights (int8 rows; about a minute)...")
    writer = wf.Writer()
    for name, quantize in wf.tensor_names(cfg.n_layer):
        t = state[name].float().contiguous()
        writer.add(name, list(t.shape), t.view(-1).tolist(), quantize)
    config = {k: getattr(cfg, k) for k in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")}
    manifest = writer.save(args.out, config, "", tok.special["<|eos|>"],
                           extra={"params": model.num_params(), "tokenizer": "tokenizer.json",
                                  "special": tok.special})
    tok.save(os.path.join(args.out, "tokenizer.json"))
    print(f"model.bin: {manifest['bytes'] / 1e6:.2f} MB ({model.num_params():,} parameters)")

    # PyTorch on the exported (int8) weights: the browser engine must give the same numbers.
    _, tensors = wf.read(args.out)
    ids = ft.prompt_ids(tok, CHECK_PASSAGE, CHECK_QUESTION, CHECK_HISTORY)
    with torch.no_grad():
        for name, (shape, flat) in tensors.items():
            state[name].copy_(torch.tensor(flat, dtype=torch.float32).view(shape))
        logits, _ = model(torch.tensor([ids]))
    positions = sorted(set(list(range(0, len(ids), max(1, len(ids) // 8))) + [len(ids) - 1]))
    with open(os.path.join(args.out, "check.json"), "w", encoding="utf-8") as f:
        json.dump({"passage": CHECK_PASSAGE, "history": CHECK_HISTORY, "question": CHECK_QUESTION,
                   "ids": ids, "positions": positions,
                   "logits": [logits[0, p].tolist() for p in positions]}, f)
    nxt = int(logits[0, -1].argmax())
    print(f"check prompt: {len(ids)} tokens; int8 model starts its answer with {tok.decode([nxt])!r}")

    pre = read_json(os.path.join(args.root, "training.json")) or {}
    fine = read_json(os.path.join(qa, "training.json")) or {}
    cal = read_json(os.path.join(qa, "calibration.json"))
    reader = {
        "params": model.num_params(),
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
        "calibration": cal,
        # How the browser should answer: decoding ("copy"/"free"), confidence kind and cut-off.
        "decoding": cal.get("recommended") if cal else None,
        "history_turns": ft.HISTORY_TURNS,
    }
    with open(os.path.join(args.out, "reader.json"), "w", encoding="utf-8") as f:
        json.dump(reader, f, indent=1)
    if not cal:
        print("note: no calibration.json yet (run evaluate.py), so reader.json has no decoding choice")
    print(f"wrote {args.out}/. Copy these files to the portfolio's public/models/tiny-gpt-reader/")


if __name__ == "__main__":
    main()
