"""Pretrain the Tiny GPT Reader on general English (Simple English Wikipedia), in word pieces.

This is step 1 of 2: here the model learns English and facts from encyclopedia text. Step 2
(fine-tuning on questions and answers) teaches it to pull answers out of a passage.

    python pretrain.py --preset smoke --out out-reader-smoke    # 2-minute check (GPU)
    python pretrain.py --out /content/drive/MyDrive/tiny-gpt-reader
    python pretrain.py --prepare-only --text-files a.txt b.txt  # data only, no torch needed

Everything is written to --out, and a run **resumes** from there: put --out on Google Drive and,
if Colab disconnects, run the same command again.
    tokenizer.json        byte-level BPE (bpe.py), trained on the first ~40 MB of articles
    train.bin, val.bin    every article as token ids (uint16), each ending in <|eos|>
    ckpt.pt               model, optimizer and progress, saved every few hundred steps
    training.json         loss curve, samples and run stats

Data preparation is pure Python (testable without a GPU); training needs PyTorch and numpy,
which Colab has.
"""

import argparse
import json
import os
import random
import time
import urllib.request
from array import array

import bpe

WIKI_URL = ("https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
            "20231101.simple/train-00000-of-00001.parquet")
SAMPLE_PROMPTS = ["The capital of France is", "Water is made of", "In 1969,", "A computer is"]

PRESETS = {
    # ~17.5M parameters; about 2 hours on a free Colab T4 (the log prints the real estimate).
    "default": dict(n_layer=8, n_head=6, n_embd=384, block_size=512, batch_size=32,
                    iters=20000, lr=6e-4, warmup=500, eval_every=500, eval_batches=20,
                    save_every=250, vocab=8192, tokenizer_mb=40, max_mb=None),
    "smoke": dict(n_layer=2, n_head=2, n_embd=64, block_size=128, batch_size=8,
                  iters=60, lr=1e-3, warmup=10, eval_every=20, eval_batches=2,
                  save_every=20, vocab=1024, tokenizer_mb=2, max_mb=4),
}


# ---------------------------------------------------------------- data (pure Python)

def download(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"downloading {url.rsplit('/', 1)[-1]}...")
    with urllib.request.urlopen(url) as r, open(path + ".part", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(path + ".part", path)


def clean(text):
    """Tidies one article: no trailing spaces, at most one blank line in a row."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    out, blank = [], False
    for line in lines:
        if not line:
            if not blank and out:
                out.append("")
            blank = True
        else:
            out.append(line)
            blank = False
    return "\n".join(out).strip()


def read_articles(text_files, max_mb):
    """Wikipedia articles (Colab: pyarrow reads the parquet file), or blank-line-separated
    documents from --text-files for testing without the download."""
    if text_files:
        docs = []
        for path in text_files:
            with open(path, encoding="utf-8", errors="replace") as f:
                docs.extend(d for d in f.read().split("\n\n") if d.strip())
    else:
        import pyarrow.parquet as pq  # Colab has it; only needed for the real data

        path = "data/simplewiki-20231101.parquet"
        download(WIKI_URL, path)
        docs = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    docs = [clean(d) for d in docs]
    docs = [d for d in docs if len(d) >= 200]  # stubs teach little
    random.Random(1337).shuffle(docs)
    if max_mb:
        kept, size = [], 0
        for d in docs:
            if size > max_mb * 1_000_000:
                break
            kept.append(d)
            size += len(d)
        docs = kept
    return docs


def write_tokens(tok, docs, path):
    eos = tok.special["<|eos|>"]
    ids = array("H")  # uint16: the vocabulary is under 65,536
    for i, doc in enumerate(docs):
        ids.extend(tok.encode(doc))
        ids.append(eos)
        if (i + 1) % 20000 == 0:
            print(f"    {i + 1:,} of {len(docs):,} articles, {len(ids):,} tokens")
    with open(path + ".part", "wb") as f:
        ids.tofile(f)
    os.replace(path + ".part", path)
    return len(ids)


def prepare(out, p, text_files):
    """Tokenizer + train.bin/val.bin in `out`, made once and reused when a run resumes."""
    tok_path = os.path.join(out, "tokenizer.json")
    train_path, val_path = os.path.join(out, "train.bin"), os.path.join(out, "val.bin")
    stats_path = os.path.join(out, "data.json")
    if all(os.path.exists(x) for x in (tok_path, train_path, val_path, stats_path)):
        print("data already prepared, reusing it")
        with open(stats_path, encoding="utf-8") as f:
            return bpe.Tokenizer.load(tok_path), json.load(f)

    os.makedirs(out, exist_ok=True)
    started = time.time()
    docs = read_articles(text_files, p["max_mb"])
    n_val = max(1, len(docs) // 200)  # 0.5% of articles, never trained on
    val_docs, train_docs = docs[:n_val], docs[n_val:]
    chars = sum(len(d) for d in docs)
    print(f"{len(docs):,} articles, {chars:,} characters ({time.time() - started:.0f} s)")

    sample, size = [], 0
    for d in train_docs:
        if size > p["tokenizer_mb"] * 1_000_000:
            break
        sample.append(d)
        size += len(d)
    tok = bpe.train(sample, p["vocab"])
    tok.save(tok_path)

    print("encoding articles into tokens...")
    n_train = write_tokens(tok, train_docs, train_path)
    n_val_tokens = write_tokens(tok, val_docs, val_path)
    stats = {
        "dataset": "Simple English Wikipedia (Wikimedia dump of 2023-11-01)" if not text_files
        else "local text files",
        "articles": len(docs),
        "characters": chars,
        "train_tokens": n_train,
        "val_tokens": n_val_tokens,
        "chars_per_token": round(chars / max(1, n_train + n_val_tokens), 2),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print(f"{n_train:,} training tokens, {n_val_tokens:,} validation tokens, "
          f"{stats['chars_per_token']} characters per token ({time.time() - started:.0f} s)")
    return tok, stats


# ---------------------------------------------------------------- training (PyTorch)

def train(out, p, tok, data_stats, device_arg):
    import math

    import numpy as np
    import torch

    from model import GPT, GPTConfig

    device = device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("warning: no GPU found. In Colab: Runtime > Change runtime type > T4 GPU.")
    train_data = np.memmap(os.path.join(out, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(out, "val.bin"), dtype=np.uint16, mode="r")
    block, batch = p["block_size"], p["batch_size"]
    assert len(val_data) > block + 1, "validation set too small for the block size"

    def get_batch(data):
        ix = np.random.randint(0, len(data) - block - 1, size=batch)
        x = torch.from_numpy(np.stack([data[i : i + block] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1 : i + 1 + block] for i in ix]).astype(np.int64))
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=block,
                    n_layer=p["n_layer"], n_head=p["n_head"], n_embd=p["n_embd"])
    model = GPT(cfg).to(device)
    opt = model.configure_optimizer(weight_decay=0.1, lr=p["lr"], betas=(0.9, 0.95), device_type=device)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def autocast():
        return torch.autocast(device, dtype=torch.float16, enabled=use_amp)

    eos = tok.special["<|eos|>"]
    ckpt_path = os.path.join(out, "ckpt.pt")
    log = {
        "dataset": data_stats["dataset"],
        "articles": data_stats["articles"],
        "characters": data_stats["characters"],
        "train_tokens": data_stats["train_tokens"],
        "chars_per_token": data_stats["chars_per_token"],
        "params": model.num_params(),
        "config": cfg.to_dict(),
        "batch_size": batch,
        "iters": p["iters"],
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "loss": [],
        "samples": [],
        "train_seconds": 0,
    }
    step = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt["config"] != cfg.to_dict():
            raise SystemExit(f"{ckpt_path} has a different model size. Use another --out folder.")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step, log = ckpt["step"], ckpt["log"]
        print(f"resuming at step {step:,} of {p['iters']:,}")
    print(f"model: {model.num_params():,} parameters, {cfg.n_layer} layers, {cfg.n_embd} wide, "
          f"{block}-token memory; {data_stats['train_tokens']:,} training tokens")

    def lr_at(s):
        if s < p["warmup"]:
            return p["lr"] * (s + 1) / p["warmup"]
        progress = (s - p["warmup"]) / max(1, p["iters"] - p["warmup"])
        return p["lr"] / 10 + 0.5 * (p["lr"] - p["lr"] / 10) * (1 + math.cos(math.pi * progress))

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        result = {}
        for name, data in (("train", train_data), ("val", val_data)):
            losses = []
            for _ in range(p["eval_batches"]):
                x, y = get_batch(data)
                with autocast():
                    _, loss = model(x, y)
                losses.append(loss.item())
            result[name] = round(sum(losses) / len(losses), 4)
        model.train()
        return result

    @torch.no_grad()
    def sample(prompt, length=60):
        model.eval()
        idx = torch.tensor([[eos] + tok.encode(prompt)], device=device)
        ids = model.generate(idx, length, temperature=0.7, top_k=40, stop_token=eos)[0].tolist()[1:]
        model.train()
        return tok.decode([i for i in ids if i != eos])

    def save(elapsed_before):
        log["steps_done"] = step
        log["train_seconds"] = round(elapsed_before + time.time() - session_start)
        with open(os.path.join(out, "training.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(), "config": cfg.to_dict(), "step": step, "log": log},
                   ckpt_path + ".part")
        os.replace(ckpt_path + ".part", ckpt_path)  # a disconnect mid-save keeps the old checkpoint

    session_start = time.time()
    elapsed_before = log["train_seconds"]
    session_first_step = step
    x, y = get_batch(train_data)
    while True:
        if step % p["eval_every"] == 0 or step == p["iters"]:
            losses = estimate_loss()
            if not log["loss"] or log["loss"][-1]["step"] != step:
                log["loss"].append({"step": step, **losses})
            texts = [sample(q) for q in SAMPLE_PROMPTS]
            if step in (0, p["eval_every"], 4 * p["eval_every"], p["iters"] // 2, p["iters"]):
                log["samples"].append({"step": step, "texts": texts})
            done = step - session_first_step
            per_step = (time.time() - session_start) / max(done, 1)
            print(f"step {step:>6} | train {losses['train']:.3f} | val {losses['val']:.3f} | "
                  f"{(elapsed_before + time.time() - session_start) / 60:.1f} min"
                  + (f", ~{per_step * (p['iters'] - step) / 60:.0f} min left" if done else ""))
            for t in texts[:2]:
                print(f"    {t[:110]!r}")
            save(elapsed_before)
        elif step % p["save_every"] == 0:
            save(elapsed_before)
        if step >= p["iters"]:
            break

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with autocast():
            _, loss = model(x, y)
        x, y = get_batch(train_data)  # fetch the next batch while the GPU works
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1

    print(f"done: {step:,} steps in {log['train_seconds'] / 60:.1f} min. "
          f"Final loss: train {log['loss'][-1]['train']}, val {log['loss'][-1]['val']}. "
          f"Next: fine-tune on questions (step 4).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="default")
    ap.add_argument("--out", default="out-reader")
    ap.add_argument("--iters", type=int, help="override the number of training steps")
    ap.add_argument("--text-files", nargs="*", help="use these files instead of Wikipedia (testing)")
    ap.add_argument("--prepare-only", action="store_true", help="make the tokenizer and token files, then stop")
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    args = ap.parse_args()

    p = dict(PRESETS[args.preset])
    if args.iters:
        p["iters"] = args.iters
    tok, stats = prepare(args.out, p, args.text_files)
    if not args.prepare_only:
        train(args.out, p, tok, stats, args.device)


if __name__ == "__main__":
    main()
