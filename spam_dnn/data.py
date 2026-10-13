"""Loading, balancing and splitting the dataset (Phase 1)."""
import pandas as pd


def load_dataset(path, text_cols=("email_text",), label_col="label"):
    """Return a DataFrame with two columns: `text` (str) and `label` (0 = ham, 1 = spam).

    text_cols: one or several text columns, joined with a space.
    A string label column such as "ham"/"spam" is mapped to 0/1, which also
    makes the SMS Spam Collection usable (text_cols=["v2"], label_col="v1").
    """
    try:
        raw = pd.read_csv(path)
    except UnicodeDecodeError:
        raw = pd.read_csv(path, encoding="latin-1")

    text = raw[list(text_cols)].fillna("").astype(str).agg(" ".join, axis=1)
    label = raw[label_col]
    if not pd.api.types.is_numeric_dtype(label):
        mapping = {"ham": 0, "spam": 1}
        label = label.astype(str).str.strip().str.lower().map(mapping)
        if label.isna().any():
            raise ValueError("Unknown label values; expected ham/spam or 0/1")

    df = pd.DataFrame({"text": text, "label": label.astype(int)})
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    return df


def class_counts(df):
    return {int(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}


def balance_undersample(df, seed=42):
    """Guarantee a 50/50 spam/ham split by undersampling the majority class."""
    n = df["label"].value_counts().min()
    parts = [g.sample(n=n, random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def balanced_split(df, test_size=0.2, seed=42):
    """Split into train/test sets that are both balanced (stratified per class)."""
    train_parts, test_parts = [], []
    for _, g in df.groupby("label"):
        g = g.sample(frac=1, random_state=seed)
        n_test = int(round(len(g) * test_size))
        test_parts.append(g.iloc[:n_test])
        train_parts.append(g.iloc[n_test:])
    train = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, test
