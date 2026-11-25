"""Pure-Python forward pass of the exported model, and a self-test model for the browser engine.

    python reference.py selftest out/selftest

writes a small random model (model.json, model.bin) plus check.json: the logits this reference
computes for every position of a test prompt. The portfolio's `npm run check:model -- <dir>` runs
the TypeScript engine on the same files and compares. It needs no torch, so it runs anywhere.
"""

import json
import math
import os
import random
import sys

import weights_format as wf

LN_EPS = 1e-5


def layer_norm(x, w, b):
    n = len(x)
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n
    inv = 1.0 / math.sqrt(var + LN_EPS)
    return [(x[i] - mean) * inv * w[i] + b[i] for i in range(n)]


def linear(W, b, x, out_dim):
    """W is [out_dim, in_dim] row-major, like torch.nn.Linear.weight."""
    n = len(x)
    out = []
    for o in range(out_dim):
        row = o * n
        s = b[o] if b is not None else 0.0
        for i in range(n):
            s += W[row + i] * x[i]
        out.append(s)
    return out


def gelu(v):
    return 0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v**3)))


def forward(manifest, t, ids):
    """Logits for every position of `ids` (a full causal pass, no cache)."""
    cfg = manifest["config"]
    C, H, L, V = cfg["n_embd"], cfg["n_head"], cfg["n_layer"], cfg["vocab_size"]
    D = C // H
    T = len(ids)
    w = lambda name: t[name][1]  # noqa: E731

    wte, wpe = w("wte.weight"), w("wpe.weight")
    xs = [[wte[tok * C + i] + wpe[p * C + i] for i in range(C)] for p, tok in enumerate(ids)]
    for l in range(L):
        p = f"h.{l}."
        hs = [layer_norm(x, w(p + "ln_1.weight"), w(p + "ln_1.bias")) for x in xs]
        qkv = [linear(w(p + "attn.c_attn.weight"), w(p + "attn.c_attn.bias"), h, 3 * C) for h in hs]
        ys = []
        for i in range(T):
            y = [0.0] * C
            for hd in range(H):
                q = qkv[i][hd * D : (hd + 1) * D]
                scores = []
                for j in range(i + 1):  # causal: only look back
                    k = qkv[j][C + hd * D : C + (hd + 1) * D]
                    scores.append(sum(q[d] * k[d] for d in range(D)) / math.sqrt(D))
                m = max(scores)
                e = [math.exp(s - m) for s in scores]
                z = sum(e)
                for j in range(i + 1):
                    a = e[j] / z
                    v = qkv[j][2 * C + hd * D : 2 * C + (hd + 1) * D]
                    for d in range(D):
                        y[hd * D + d] += a * v[d]
            ys.append(y)
        proj = [linear(w(p + "attn.c_proj.weight"), w(p + "attn.c_proj.bias"), y, C) for y in ys]
        xs = [[xs[i][c] + proj[i][c] for c in range(C)] for i in range(T)]
        hs = [layer_norm(x, w(p + "ln_2.weight"), w(p + "ln_2.bias")) for x in xs]
        for i in range(T):
            f = [gelu(v) for v in linear(w(p + "mlp.c_fc.weight"), w(p + "mlp.c_fc.bias"), hs[i], 4 * C)]
            out = linear(w(p + "mlp.c_proj.weight"), w(p + "mlp.c_proj.bias"), f, C)
            xs[i] = [xs[i][c] + out[c] for c in range(C)]
    xs = [layer_norm(x, w("ln_f.weight"), w("ln_f.bias")) for x in xs]
    return [linear(wte, None, x, V) for x in xs]  # tied output layer


def selftest(out_dir):
    rng = random.Random(1234)
    vocab = "\x00\n !',.?abcdefghijklmnopqrstuvwxyz"
    cfg = {"vocab_size": len(vocab), "block_size": 16, "n_layer": 2, "n_head": 2, "n_embd": 16}
    writer = wf.Writer()
    C = cfg["n_embd"]
    shapes = {
        "wte.weight": [cfg["vocab_size"], C],
        "wpe.weight": [cfg["block_size"], C],
        "ln_1.weight": [C], "ln_1.bias": [C],
        "attn.c_attn.weight": [3 * C, C], "attn.c_attn.bias": [3 * C],
        "attn.c_proj.weight": [C, C], "attn.c_proj.bias": [C],
        "ln_2.weight": [C], "ln_2.bias": [C],
        "mlp.c_fc.weight": [4 * C, C], "mlp.c_fc.bias": [4 * C],
        "mlp.c_proj.weight": [C, 4 * C], "mlp.c_proj.bias": [C],
        "ln_f.weight": [C], "ln_f.bias": [C],
    }
    for name, quantize in wf.tensor_names(cfg["n_layer"]):
        shape = shapes[name.split(".", 2)[2] if name.startswith("h.") else name]
        n = math.prod(shape)
        if name.endswith("ln_1.weight") or name.endswith("ln_2.weight") or name == "ln_f.weight":
            flat = [1.0 + rng.gauss(0, 0.1) for _ in range(n)]
        else:
            flat = [rng.gauss(0, 0.5) for _ in range(n)]
        writer.add(name, shape, flat, quantize)
    writer.save(out_dir, cfg, vocab, eos_id=0, extra={"params": 0, "selftest": True})

    manifest, tensors = wf.read(out_dir)
    prompt = "once upon a time"[: cfg["block_size"]]
    ids = [vocab.index(ch) for ch in prompt]
    logits = forward(manifest, tensors, ids)
    with open(os.path.join(out_dir, "check.json"), "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "ids": ids, "logits": logits}, f)
    print(f"wrote self-test model to {out_dir} ({len(ids)} positions checked)")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "selftest":
        selftest(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)
