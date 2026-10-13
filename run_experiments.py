"""Required Work Unit 1: One-Hot vs BoW vs TF-IDF on a very deep dense network.

Pipeline: load -> balance 50/50 -> balanced train/test split -> vocabulary and
IDF from the training set only -> vectorise (3 ways) -> build a 41-hidden-layer
network with He initialisation -> train with BCE for 1000 epochs -> evaluate.

Examples
  python run_experiments.py                        # full run (defaults)
  python run_experiments.py --epochs 20 --layers 5 # quick smoke test
  python run_experiments.py --compare-init         # also train with PyTorch's default init
  # SMS Spam Collection instead of the current dataset:
  python run_experiments.py --data data/spam.csv --text-cols v2 --label-col v1
"""
import argparse
import json
import os
import platform
import time

import pandas as pd
import torch

from spam_dnn import plots
from spam_dnn.data import balance_undersample, balanced_split, class_counts, load_dataset
from spam_dnn.evaluate import compute_metrics
from spam_dnn.model import build_deep_dnn
from spam_dnn.train import predict, train_model
from spam_dnn.vectorizers import (bow_vectorize, build_vocabulary, compute_idf,
                                  one_hot_vectorize, tfidf_vectorize)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/spam_email_dataset.csv")
    p.add_argument("--text-cols", nargs="+", default=["email_text"],
                   help="text column(s) to vectorise (default: email_text only)")
    p.add_argument("--label-col", default="label")
    p.add_argument("--vocab-size", type=int, default=500, help="N = input dimension")
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--layers", type=int, default=41, help="number of hidden layers L")
    p.add_argument("--width", type=int, default=64, help="neurons per hidden layer")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results")
    p.add_argument("--compare-init", action="store_true",
                   help="also train every model with PyTorch's default initialisation")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- Phase 1: data -------------------------------------------------
    df = load_dataset(args.data, args.text_cols, args.label_col)
    print("Raw class counts (0 = ham, 1 = spam):", class_counts(df))
    balanced = balance_undersample(df, seed=args.seed)
    train_df, test_df = balanced_split(balanced, test_size=args.test_size, seed=args.seed)
    print("After balancing:", class_counts(balanced))
    print("Train:", class_counts(train_df), "| Test:", class_counts(test_df))

    # vocabulary and IDF statistics: training set only
    vocab = build_vocabulary(train_df["text"], max_size=args.vocab_size, min_df=args.min_df)
    idf = compute_idf(train_df["text"], vocab)
    print(f"Vocabulary size (input dimension): {len(vocab)}")

    tr, te = list(train_df["text"]), list(test_df["text"])
    y_train, y_test = train_df["label"].to_numpy(), test_df["label"].to_numpy()
    features = {
        "One-Hot": (one_hot_vectorize(tr, vocab), one_hot_vectorize(te, vocab)),
        "BoW": (bow_vectorize(tr, vocab), bow_vectorize(te, vocab)),
        "TF-IDF": (tfidf_vectorize(tr, vocab, idf), tfidf_vectorize(te, vocab, idf)),
    }

    # ---- Phases 2 and 3: model, training, evaluation ---------------------
    inits = ["he"] + (["default"] if args.compare_init else [])
    histories, metrics = {}, {}
    for init in inits:
        for name, (X_tr, X_te) in features.items():
            run = name if init == "he" else f"{name} (default init)"
            print(f"\n=== {run}: {args.layers} hidden layers x {args.width} neurons, "
                  f"input {X_tr.shape[1]}, {args.epochs} epochs ===")
            torch.manual_seed(args.seed)
            model = build_deep_dnn(X_tr.shape[1], args.layers, args.width, init=init)
            t0 = time.time()
            histories[run] = train_model(model, X_tr, y_train, epochs=args.epochs, lr=args.lr)
            metrics[run] = compute_metrics(y_test, predict(model, X_te))
            metrics[run]["final_loss"] = histories[run]["loss"][-1]
            metrics[run]["diverged"] = histories[run]["diverged"]
            m = metrics[run]
            print(f"  test acc {m['accuracy']:.4f} | precision {m['precision']:.4f} | "
                  f"recall {m['recall']:.4f} | F1 {m['f1']:.4f} | FPR {m['fpr']:.4f} "
                  f"| {time.time() - t0:.0f}s")

    # ---- outputs ---------------------------------------------------------
    table = pd.DataFrame(metrics).T
    cols = ["accuracy", "precision", "recall", "f1", "fpr", "tn", "fp", "fn", "tp",
            "final_loss", "diverged"]
    table = table[cols]
    table.index.name = "run"
    table.to_csv(os.path.join(args.out, "metrics.csv"))

    lines = ["| Run | Accuracy | Precision | Recall | F1 | FPR | TN | FP | FN | TP | Final loss |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, r in table.iterrows():
        lines.append(f"| {run} | {r.accuracy:.4f} | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} "
                     f"| {r.fpr:.4f} | {int(r.tn)} | {int(r.fp)} | {int(r.fn)} | {int(r.tp)} "
                     f"| {r.final_loss:.4f} |")
    with open(os.path.join(args.out, "metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    with open(os.path.join(args.out, "histories.json"), "w") as f:
        json.dump({run: {"loss": h["loss"], "train_acc": h["train_acc"],
                         "diverged": h["diverged"],
                         "grad_norms": {str(e): g for e, g in h["grad_norms"].items()}}
                   for run, h in histories.items()}, f)

    config = vars(args) | {
        "n_train": len(train_df), "n_test": len(test_df), "vocab_size": len(vocab),
        "python": platform.python_version(), "torch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "platform": platform.platform(),
    }
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    plots.plot_loss_curves(histories, os.path.join(args.out, "loss_curves.png"))
    plots.plot_grad_flow(histories, os.path.join(args.out, "gradient_flow.png"))
    plots.plot_confusion_matrices(metrics, os.path.join(args.out, "confusion_matrices.png"))
    print(f"\nSaved metrics, histories, run_config.json and 3 figures to {args.out}/")


if __name__ == "__main__":
    main()
