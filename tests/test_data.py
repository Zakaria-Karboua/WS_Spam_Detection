import pandas as pd

from spam_dnn.data import balance_undersample, balanced_split, class_counts


def _imbalanced(n_ham=87, n_spam=13):
    return pd.DataFrame({
        "text": [f"ham {i}" for i in range(n_ham)] + [f"spam {i}" for i in range(n_spam)],
        "label": [0] * n_ham + [1] * n_spam,
    })


def test_balance_gives_50_50():
    out = balance_undersample(_imbalanced())
    assert class_counts(out) == {0: 13, 1: 13}


def test_split_is_balanced_and_disjoint():
    bal = balance_undersample(_imbalanced(870, 130))
    train, test = balanced_split(bal, test_size=0.2)
    assert class_counts(test) == {0: 26, 1: 26}
    assert class_counts(train) == {0: 104, 1: 104}
    assert set(train["text"]).isdisjoint(set(test["text"]))
