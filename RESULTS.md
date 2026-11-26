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
| 2026-11-26 | **Pointing (span) Reader**, same pretraining (`finetune_span.py`) | 10,000 / 16 min, T4 | **53.7 / 57.6** | 71.7% | 43.3% (40.7%) | **48.4 / 58.8** | 98.6% (49.0%) | +11.7 / +13.3 exact over writing the answer, from the same model. SQuAD exam 11,821 questions |

### Confidence cut-off (evaluate.py)

The chat shows the Reader's answer only above this cut-off; below it, the source sentence.
Target: shown answers exactly right at least 80% of the time.

| Date | Decoding | Questions scored | Best cut-off | Shown | Right among shown | Notes |
|---|---|---|---|---|---|---|
| 2026-11-26 | Free (greedy) | 1,500 SQuAD + 1,500 CoQA (CPU) | none reaches 80% | at 0.95: 4.4% | 72.7% | Sample scores match the full exam (SQuAD 41.3 vs 42.0, CoQA 35.5 vs 35.1). At 0.8: 30.1% shown, 47.7% right |
| 2026-11-26 | Forced copying | 1,500 SQuAD + 1,500 CoQA (CPU) | none reaches 80% | at 0.95: 4.4% (raw) / 8.4% (within allowed) | 72.7% / 61.1% | Barely differs from free: the Reader already copies; its mistakes are picking the wrong phrase. "Useful" (F1 ≥ 0.5) only ~7 points above exact |
| 2026-11-26 | Free + agreement check | 1,500 SQuAD + 1,500 CoQA (CPU) | none reaches 80% | at 0.9: 5.6% | 62.7% | Only 44.9% of span answers lie inside the search's sentence; agreement halves coverage for ~4 points. Yes/no alone: 65.1% right overall, 80.8% at 0.9 (1.7% shown) |
| 2026-11-26 | **Pointing (span) Reader** | every question (11,821 SQuAD + 7,962 CoQA, GPU) | **0.6** | 8.9% | **82.2%** (84.8% useful) | First setting to pass 80%. At 0.5: 13.2% shown, 79.1% right; at 0.4: 19.2% shown, 74.1% right. Confidence now tracks correctness. Agreement check adds nothing (0.6: 5.7% shown, 81.0% right): only 61% of answers lie in the search's sentence. Yes/no alone: 81% right at 0.5 |
