"""Three ways to turn raw text into fixed-size numeric vectors (Phase 1).

The vocabulary and the IDF statistics are computed on the TRAINING texts only
and then reused, unchanged, for the test texts (no data leakage).
"""
import re
from collections import Counter

import numpy as np

# URLs, words/numbers, and a few meaningful symbols (money signs, percent)
_TOKEN_RE = re.compile(r"https?://\S+|www\.\S+|[a-z0-9]+|[£$€%]")


def tokenize(text):
    """Lowercase, drop punctuation/whitespace, keep words, numbers, symbols, URLs."""
    return _TOKEN_RE.findall(str(text).lower())


def build_vocabulary(train_texts, max_size=500, min_df=2):
    """Global vocabulary from the training corpus: the `max_size` tokens that
    appear in the most messages (document frequency), ignoring tokens seen in
    fewer than `min_df` messages. Returns {token: column_index}.

    The input dimension of the network is len(vocab).
    """
    df = Counter()
    for text in train_texts:
        df.update(set(tokenize(text)))
    items = [(tok, c) for tok, c in df.items() if c >= min_df]
    items.sort(key=lambda x: (-x[1], x[0]))          # deterministic order
    return {tok: i for i, (tok, _) in enumerate(items[:max_size])}


def _count_matrix(texts, vocab):
    """Raw token counts (n_messages, V) and total token count per message."""
    X = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    lengths = np.zeros(len(texts), dtype=np.float32)
    for i, text in enumerate(texts):
        tokens = tokenize(text)
        lengths[i] = len(tokens)
        for tok in tokens:
            j = vocab.get(tok)
            if j is not None:
                X[i, j] += 1.0
    return X, lengths


def one_hot_vectorize(texts, vocab):
    """Binary presence: 1 if the token occurs in the message, else 0."""
    counts, _ = _count_matrix(texts, vocab)
    return (counts > 0).astype(np.float32)


def bow_vectorize(texts, vocab):
    """Bag of Words: number of occurrences of each vocabulary token."""
    counts, _ = _count_matrix(texts, vocab)
    return counts


def compute_idf(train_texts, vocab):
    """idf_t = log(N / DF_t), with N and DF_t measured on the training set."""
    counts, _ = _count_matrix(train_texts, vocab)
    n_docs = counts.shape[0]
    df = np.maximum((counts > 0).sum(axis=0), 1)
    return np.log(n_docs / df).astype(np.float32)


def tfidf_vectorize(texts, vocab, idf):
    """W(t, d) = TF(t, d) * log(N / DF_t), TF = count / number of tokens in d."""
    counts, lengths = _count_matrix(texts, vocab)
    tf = counts / np.maximum(lengths, 1.0)[:, None]
    return (tf * idf[None, :]).astype(np.float32)
