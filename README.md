# Spam Detection with a Very Deep Dense Network in PyTorch

Workshop module · ENSM · Unit 1 (Dr. Amel Bouchemha)

This repository contains two pieces of work:

1. **Practical example** (`notebooks/01_spam_practical_example.ipynb`): a small ANN trained on 16 hand-written emails with 10 keyword features, plus five experiments.
2. **Required work** (`spam_dnn/` + `run_experiments.py`): spam detection on a Kaggle dataset with a dense network of **41 hidden layers**, comparing three text vectorizations (**One-Hot, Bag of Words, TF-IDF**) with **He initialization**, **Binary Cross-Entropy** and **1000 epochs**.

## Required work: how each requirement is covered

| Requirement | Where |
|---|---|
| Three vectorizers: One-Hot, BoW, TF-IDF | `spam_dnn/vectorizers.py` |
| Fixed output size (input dimension N) | `build_vocabulary(max_size=N)`, option `--vocab-size` |
| Vocabulary and IDF from the training set only | `build_vocabulary`, `compute_idf` (called on the train split only) |
| 50/50 balancing (undersampling) | `spam_dnn/data.py: balance_undersample` |
| Balanced train and test sets | `spam_dnn/data.py: balanced_split` |
| Input layer matches the vector size | `run_experiments.py` passes `X.shape[1]` to the model |
| More than 40 hidden layers built in a loop | `spam_dnn/model.py` (default 41, option `--layers`) |
| Neurons per layer | option `--width` (default 64) |
| He initialization | `spam_dnn/model.py` (`init="he"`) |
| BCE loss, 1000+ epochs | `spam_dnn/train.py` (`BCEWithLogitsLoss`, default 1000) |
| Loss curves for the 3 vectorizations | `results/loss_curves.png` |
| Gradient flow across layers | `results/gradient_flow.png` |
| Accuracy, precision, recall, F1, confusion matrix, false-positive rate | `spam_dnn/evaluate.py`, `results/metrics.md`, `results/confusion_matrices.png` |
| Project report | `report/REPORT.md` |

## Quick start

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate

pip install -r requirements.txt
python run_experiments.py --epochs 20 --layers 5   # quick smoke test (seconds)
python run_experiments.py                          # full run: 41 layers, 1000 epochs
python run_experiments.py --compare-init           # also train with PyTorch's default init
```

Outputs are written to `results/`: `metrics.csv`, `metrics.md`, `histories.json`, `run_config.json` (hyperparameters and environment for the report) and three figures.

Useful options: `--vocab-size`, `--layers`, `--width`, `--epochs`, `--lr`, `--seed`, `--text-cols`, `--label-col`. Run `python run_experiments.py --help` for all of them.

If the loss curves look flat or unstable, adjust `--lr` (the default is 1e-4).

Tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## Dataset

Kaggle: *Spam Email Detection Dataset (Clean & ML Ready)*. 10,000 emails, 20 columns, 6,005 ham and 3,995 spam. Place the CSV in `data/` (see `data/README.md`); it is git-ignored.

Things worth knowing before reading the results:

* Only the raw message (`email_text`) is vectorized. The columns `subject` and `sender_domain` each belong to exactly one class (8 distinct values each, none shared between ham and spam), so using them would make the task trivial and would not be learning from the message text.
* The text looks synthetic: spam messages start with a stretch of spam keywords (win, free, urgent, offer, limited, click, now, cash, guarantee) followed by random filler words. A plain logistic regression on any of the three vectorizations separates the balanced test set perfectly, so near-perfect scores are expected. The interesting differences are in **convergence and gradient flow**, not in accuracy.
* The class imbalance is mild (60/40), not the 87/13 of the SMS Spam Collection. The balancing routine still runs and gives 3,995 messages per class.
* The pipeline is dataset-agnostic. For a harder, more realistic benchmark, use the SMS Spam Collection (see `data/README.md`).

## Results

> Fill in after running `python run_experiments.py`. The full table is in `results/metrics.md`.

| Vectorization | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| One-Hot | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| BoW | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| TF-IDF | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## Project structure

```
spam-detection-pytorch/
├── spam_dnn/            data.py · vectorizers.py · model.py · train.py · evaluate.py · plots.py
├── run_experiments.py   full pipeline for the required work
├── notebooks/           01_spam_practical_example.ipynb
├── unit1_ann_basics/    code from the slides and the "Try it" exercises
├── tests/               pytest checks (data, vectorizers, metrics, model)
├── report/REPORT.md     report skeleton
├── data/                Kaggle CSV (git-ignored)
└── results/             generated tables and figures
```

## License

MIT
