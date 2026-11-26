# Tiny GPT

A small GPT (decoder-only transformer) built and trained from scratch. It reads and writes
English one character at a time, trained on short children's stories plus addition and
subtraction, and runs in the browser on my portfolio with a hand-written TypeScript engine:
no server, no API, no ML library.

| | |
|---|---|
| Parameters | about 4.8M (6 layers, 8 heads, 256 wide) |
| Context | 256 characters |
| Data | first 200 MB of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (Eldan & Li, 2023), plus 4M generated problems like `347+28=375` and `28-347=-319` (numbers 0–999) |
| Math exam | 2,000 problems held out of training, answered greedily; score in `training.json` |
| Training | 15,000 steps, batch 64 × 256 characters, on a free Colab T4 GPU |
| Download | about 5 MB (int8 weights with one float32 scale per row) |

## Files

- `RESULTS.md`: every training run and its exam scores, so progress is easy to compare.

- `model.py`: the GPT in PyTorch.
- `bpe.py`: the byte-level BPE tokenizer for the coming Tiny GPT Reader (word pieces instead of
  characters). Pure Python; `python bpe.py test` tests it. The portfolio has a TypeScript copy
  that `npm run check:bpe` compares against test cases from `python bpe.py vectors`.
- `arithmetic.py`: makes the math problems and the held-out exam. Pure Python; `python arithmetic.py` tests it.
- `train.py`: downloads the stories, mixes in the math, trains, and logs loss, math score and
  sample stories to `out/training.json`.
- `export.py`: writes `out/web/` for the browser, plus `check.json` to verify the browser engine.
- `weights_format.py`: the `model.json` + `model.bin` format. Pure Python.
- `reference.py`: a pure-Python forward pass, plus a random self-test model for the engine.
- `colab.ipynb`: run everything on Google Colab.
- `pretrain.py`: step 1 of the Tiny GPT Reader: a ~17.5M-parameter model that reads word pieces,
  pretrained on Simple English Wikipedia. Saves and resumes from its `--out` folder.
- `finetune.py`: step 2 of the Reader: fine-tunes it on SQuAD 2.0 and CoQA to answer questions
  from a passage in a conversation (follow-ups, yes/no, or "no answer"), and scores it on both
  dev sets.
  `python finetune.py --check-data` tests the data side without a GPU.
- `evaluate.py`: scores the fine-tuned Reader with a confidence for each answer, free or forced to
  copy from the passage, and recommends how the portfolio should decode and which cut-off to use
  (shown answers exactly right at least 80% of the time). Runs on CPU.
- `export_reader.py`: writes the Reader's browser files (weights, tokenizer, engine check,
  and `reader.json` with the exam scores and cut-off).
- `reader.ipynb`: run every Reader step on Colab, with progress saved to Google Drive.

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
