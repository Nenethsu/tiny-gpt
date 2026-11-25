"""Train the tiny character-level GPT on TinyStories.

    python train.py                 # full run (GPU; about 15-30 min on a Colab T4)
    python train.py --preset smoke  # 30 steps on a tiny model, to check that everything runs

Writes to out/:
    ckpt.pt        the trained weights, config and vocabulary
    training.json  loss curve, text samples during training and run stats (shown on the portfolio page)
"""

import argparse
import json
import math
import os
import time
import urllib.request

import torch

from model import GPT, GPTConfig

DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{split}.txt"
EOS = "\x00"  # one character marks the end of each story; the browser stops there
SAMPLE_PROMPT = "Once upon a time"

# Fancy punctuation in the stories, mapped to plain ASCII so the vocabulary stays small.
REPLACE = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}
ALLOWED = set(chr(c) for c in range(32, 127)) | {"\n", EOS}

PRESETS = {
    "default": dict(n_layer=6, n_head=8, n_embd=256, block_size=256, batch_size=64,
                    iters=10000, eval_every=250, eval_batches=40, data_mb=200),
    "smoke": dict(n_layer=2, n_head=2, n_embd=32, block_size=64, batch_size=8,
                  iters=30, eval_every=10, eval_batches=2, data_mb=2),
}


def download(split, max_bytes, path):
    """Fetches the first `max_bytes` of a TinyStories split (the full train file is 2 GB)."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading {split} ({max_bytes / 1e6:.0f} MB)...")
    req = urllib.request.Request(DATA_URL.format(split=split), headers={"Range": f"bytes=0-{max_bytes - 1}"})
    with urllib.request.urlopen(req) as r, open(path + ".part", "wb") as f:
        left = max_bytes
        while left > 0:
            chunk = r.read(min(1 << 20, left))
            if not chunk:
                break
            f.write(chunk)
            left -= len(chunk)
    os.replace(path + ".part", path)


def clean(raw):
    """Splits into whole stories, keeps the ones made of plain characters, joins them with EOS."""
    text = raw.decode("utf-8", errors="ignore")
    for a, b in REPLACE.items():
        text = text.replace(a, b)
    stories = text.split("<|endoftext|>")[:-1]  # the last piece was cut off mid-story
    kept = []
    for story in stories:
        story = story.strip()
        if story and set(story) <= ALLOWED:
            kept.append(story)
    return "".join(s + EOS for s in kept), len(kept), len(stories)


def load_data(data_mb, block_size):
    val_mb = max(1, data_mb // 20)
    train_path, val_path = f"data/train-{data_mb}mb.txt", f"data/valid-{val_mb}mb.txt"
    download("train", data_mb * 1_000_000, train_path)
    download("valid", val_mb * 1_000_000, val_path)
    with open(train_path, "rb") as f:
        train_text, kept, total = clean(f.read())
    with open(val_path, "rb") as f:
        val_text, _, _ = clean(f.read())
    vocab = "".join(sorted(set(train_text)))
    print(f"{kept:,} of {total:,} stories kept, {len(train_text):,} characters, vocabulary of {len(vocab)}")
    table = {ord(ch): i for i, ch in enumerate(vocab)}
    val_text = "".join(ch for ch in val_text if ord(ch) in table)

    def encode(s):
        # Each character becomes one byte holding its id (the vocabulary is under 256), kept as uint8 to save RAM.
        return torch.frombuffer(bytearray(s.translate(table).encode("latin-1")), dtype=torch.uint8)

    train, val = encode(train_text), encode(val_text)
    assert len(val) > block_size + 1, "validation set too small"
    return train, val, vocab, len(train_text)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).long()
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix]).long()
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def lr_at(step, iters, max_lr, min_lr, warmup):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, iters - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="default")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--data-mb", type=int)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    p = dict(PRESETS[args.preset])
    if args.iters:
        p["iters"] = args.iters
    if args.data_mb:
        p["data_mb"] = args.data_mb
    torch.manual_seed(args.seed)
    device = args.device
    if device == "cpu" and args.preset == "default":
        print("warning: no GPU found. In Colab: Runtime > Change runtime type > T4 GPU.")

    train, val, vocab, n_chars = load_data(p["data_mb"], p["block_size"])
    cfg = GPTConfig(vocab_size=len(vocab), block_size=p["block_size"],
                    n_layer=p["n_layer"], n_head=p["n_head"], n_embd=p["n_embd"])
    model = GPT(cfg).to(device)
    print(f"model: {model.num_params():,} parameters")

    opt = model.configure_optimizer(weight_decay=0.1, lr=args.lr, betas=(0.9, 0.99), device_type=device)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = lambda: torch.autocast(device, dtype=torch.float16, enabled=use_amp)  # noqa: E731
    stoi = {ch: i for i, ch in enumerate(vocab)}
    eos_id = stoi[EOS]

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for name, data in (("train", train), ("val", val)):
            losses = torch.zeros(p["eval_batches"])
            for k in range(p["eval_batches"]):
                x, y = get_batch(data, cfg.block_size, p["batch_size"], device)
                with autocast():
                    _, loss = model(x, y)
                losses[k] = loss.item()
            out[name] = round(losses.mean().item(), 4)
        model.train()
        return out

    @torch.no_grad()
    def sample(length=300):
        model.eval()
        idx = torch.tensor([[stoi[c] for c in SAMPLE_PROMPT]], device=device)
        out = model.generate(idx, length, temperature=0.8, top_k=40, stop_token=eos_id)[0].tolist()
        model.train()
        return "".join(vocab[i] for i in out).split(EOS)[0]

    os.makedirs(args.out, exist_ok=True)
    log = {
        "dataset": "TinyStories (Eldan & Li, 2023)",
        "characters": n_chars,
        "params": model.num_params(),
        "config": cfg.to_dict(),
        "batch_size": p["batch_size"],
        "iters": p["iters"],
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "loss": [],
        "samples": [],
    }

    def save(step, elapsed):
        log["steps_done"] = step
        log["train_seconds"] = round(elapsed)
        with open(os.path.join(args.out, "training.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "vocab": vocab, "eos": EOS},
                   os.path.join(args.out, "ckpt.pt"))

    start = time.time()
    x, y = get_batch(train, cfg.block_size, p["batch_size"], device)
    for step in range(p["iters"] + 1):
        if step % p["eval_every"] == 0 or step == p["iters"]:
            losses = estimate_loss()
            log["loss"].append({"step": step, **losses})
            text = sample()
            # Keep a handful of samples to show how the writing improves (first, early, middle, last).
            if step in (0, p["eval_every"], 4 * p["eval_every"]) or step == p["iters"] or step == p["iters"] // 2:
                log["samples"].append({"step": step, "text": text})
            elapsed = time.time() - start
            eta = elapsed / max(step, 1) * (p["iters"] - step)
            print(f"step {step:>5} | train {losses['train']:.3f} | val {losses['val']:.3f} "
                  f"| {elapsed / 60:.1f} min, ~{eta / 60:.0f} min left\n    {text[:120]!r}")
            save(step, elapsed)
        if step == p["iters"]:
            break

        for g in opt.param_groups:
            g["lr"] = lr_at(step, p["iters"], args.lr, args.lr / 10, warmup=min(200, p["iters"] // 10 + 1))
        with autocast():
            _, loss = model(x, y)
        x, y = get_batch(train, cfg.block_size, p["batch_size"], device)  # fetch the next batch meanwhile
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

    print(f"done in {(time.time() - start) / 60:.1f} min. Next: python export.py")


if __name__ == "__main__":
    main()
