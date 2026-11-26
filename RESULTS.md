# Results

One row per training run, newest last. Scores come from data the model never trained on:
held-out text for loss, the fixed exam sets (same questions and seeds every run) for accuracy.
Only the final exam counts; the progress lines during training score 300 questions and swing
by several points.

Rules: change one thing per run, and write down what changed.

## Tiny GPT (characters: stories and sums)

| Date | Run | Params | Steps / time | Val loss (per character) | Math exam (2,000 sums) | Notes |
|---|---|---|---|---|---|---|
| 2026-11-25 | Stories only | 4.83M | 10,000 / 13 min, T4 | 0.577 | – | First model on the site |
| 2026-11-25 | Stories + 4M sums | 4.83M | 15,000 / 20 min, T4 | 0.615 | **98.1%** (add 99.3, sub 96.9) | Loss rose a little: it now also learns math |

## Tiny GPT Reader (word pieces: answering from a passage)

Loss here is per word piece (about 3.2 characters), so it can't be compared with the
character model's.

### Pretraining (Simple English Wikipedia)

| Date | Run | Params | Data | Steps / time | Val loss (per token) | Notes |
|---|---|---|---|---|---|---|
| 2026-11-25 | 8 layers, 384 wide, 512 tokens, 8,192 vocab | 17.54M | 195,871 articles, 81.0M tokens | 20,000 (≈4 passes) / 63 min, T4 | **2.64** (best 2.57) | Train/val gap steady at 0.3–0.4: no overfitting |

### Fine-tuning (SQuAD 2.0 + CoQA) and exams

SQuAD dev: 11,827 questions. CoQA dev: 7,962 conversation turns, asked with their history.
"Shown" = passes the safety check (word for word in the passage, or yes/no); "right" = share of
shown answers that are exactly right.

| Date | Run | Steps / time | SQuAD exact / F1 | SQuAD no-answer | SQuAD shown (right) | CoQA exact / F1 | CoQA shown (right) | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-11-26 | From the pretraining above | 10,000 / 21 min, T4 | **42.0 / 44.1** | 66.2% | 38.0% (22.8%) | **35.1 / 41.7** | 93.2% (37.5%) | Flat after ~7,000 steps. Answerable SQuAD questions: about 1 in 6 exactly right |

### Confidence cut-off (evaluate.py)

The chat shows the Reader's answer only above this cut-off; below it, the source sentence.
Target: shown answers exactly right at least 80% of the time.

| Date | Decoding | Questions scored | Best cut-off | Shown | Right among shown | Notes |
|---|---|---|---|---|---|---|
| 2026-11-26 | Free (greedy) | 1,500 SQuAD + 1,500 CoQA (CPU) | none reaches 80% | at 0.95: 4.4% | 72.7% | Sample scores match the full exam (SQuAD 41.3 vs 42.0, CoQA 35.5 vs 35.1). At 0.8: 30.1% shown, 47.7% right |
