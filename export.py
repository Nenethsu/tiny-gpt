"""Export the trained model for the browser.

    python export.py              # reads out/ckpt.pt, writes out/web/

out/web/ holds everything the portfolio needs; copy its files to public/models/tiny-gpt/:
    model.json     config, vocabulary and tensor table
    model.bin      the weights (int8 matrices, float32 biases)
    check.json     logits PyTorch computes from the exported weights, for `npm run check:model`
    training.json  loss curve, samples and run stats for the page
"""

import argparse
import json
import os
import shutil

import torch

import weights_format as wf
from model import GPT, GPTConfig

CHECK_PROMPT = "Once upon a time, there was a little dog named Max."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--out", default="out/web")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    vocab, eos = ckpt["vocab"], ckpt["eos"]
    state = model.state_dict()

    writer = wf.Writer()
    for name, quantize in wf.tensor_names(cfg.n_layer):
        t = state[name].float().contiguous()
        writer.add(name, list(t.shape), t.view(-1).tolist(), quantize)
    config = {k: getattr(cfg, k) for k in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd")}
    manifest = writer.save(args.out, config, vocab, vocab.index(eos), extra={"params": model.num_params()})
    print(f"model.bin: {manifest['bytes'] / 1e6:.2f} MB ({model.num_params():,} parameters)")

    # Read the files back and run PyTorch on those (quantized) weights. If the browser engine
    # gives the same numbers, the export and the engine agree.
    _, tensors = wf.read(args.out)
    with torch.no_grad():
        for name, (shape, flat) in tensors.items():
            state[name].copy_(torch.tensor(flat, dtype=torch.float32).view(shape))
        ids = [vocab.index(ch) for ch in CHECK_PROMPT[: cfg.block_size]]
        logits, _ = model(torch.tensor([ids]))
    with open(os.path.join(args.out, "check.json"), "w", encoding="utf-8") as f:
        json.dump({"prompt": CHECK_PROMPT, "ids": ids, "logits": logits[0].tolist()}, f)

    # Quantization costs a little accuracy; show how much.
    full = GPT(cfg)
    full.load_state_dict(ckpt["model"])
    full.eval()
    with torch.no_grad():
        x = torch.tensor([ids[:-1]])
        y = torch.tensor([ids[1:]])
        _, loss_full = full(x, y)
        _, loss_q = model(x, y)
    print(f"loss on the check prompt: float32 {loss_full.item():.4f}, int8 {loss_q.item():.4f}")

    log_path = os.path.join(os.path.dirname(args.ckpt), "training.json")
    if os.path.exists(log_path):
        shutil.copy(log_path, os.path.join(args.out, "training.json"))
    print(f"wrote {args.out}/. Copy these files to the portfolio's public/models/tiny-gpt/")


if __name__ == "__main__":
    main()
