# Tiny GPT

A small GPT (decoder-only transformer) built and trained from scratch. It reads and writes
English one character at a time, trained on short children's stories, and runs in the browser
on my portfolio with a hand-written TypeScript engine: no server, no API, no ML library.

| | |
|---|---|
| Parameters | about 4.8M (6 layers, 8 heads, 256 wide) |
| Context | 256 characters |
| Data | first 200 MB of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (Eldan & Li, 2023) |
| Training | 10,000 steps, batch 64 × 256 characters, on a free Colab T4 GPU |
| Download | about 5 MB (int8 weights with one float32 scale per row) |

## Files

- `model.py`: the GPT in PyTorch.
- `train.py`: downloads the data, trains, logs loss and sample stories to `out/training.json`.
- `export.py`: writes `out/web/` for the browser, plus `check.json` to verify the browser engine.
- `weights_format.py`: the `model.json` + `model.bin` format. Pure Python.
- `reference.py`: a pure-Python forward pass, plus a random self-test model for the engine.
- `colab.ipynb`: run everything on Google Colab.

## Train

Open `colab.ipynb` in Google Colab, choose a T4 GPU runtime and run the cells. They need
PyTorch, which Colab already has. The last cell downloads `tiny-gpt-web.zip`.

Locally, with a GPU and PyTorch installed:

```
python train.py --preset smoke   # 30 steps, checks that everything runs
python train.py
python export.py
```

## Put it on the portfolio

1. Unzip into `My-Portfolio-V2/public/models/tiny-gpt/`.
2. In the portfolio, run `npm run check:model`. It runs the TypeScript engine on the same
   prompt as PyTorch and fails if the logits differ.
3. Open `/lab`.

## Test the browser engine without training

```
python reference.py selftest out/selftest
# in the portfolio:
npm run check:model -- ../tiny-gpt/out/selftest
```

## License note

TinyStories is published under CDLA-Sharing-1.0. Check its dataset card before publishing the
trained weights.
