import numpy as np

from spam_dnn.vectorizers import (bow_vectorize, build_vocabulary, compute_idf,
                                  one_hot_vectorize, tfidf_vectorize, tokenize)

TRAIN = ["Win FREE cash now!", "free free offer, click now", "meeting at noon", "report due, thanks"]


def test_tokenize_keeps_words_numbers_symbols_urls():
    assert tokenize("Win $100 at http://x.co NOW!!") == ["win", "$", "100", "at", "http://x.co", "now"]


def test_vocabulary_comes_from_training_only_and_is_capped():
    vocab = build_vocabulary(TRAIN, max_size=3, min_df=1)
    assert len(vocab) == 3
    # a test-only word never enters the vocabulary
    assert "prize" not in build_vocabulary(TRAIN, max_size=100, min_df=1)


def test_fixed_output_size():
    vocab = build_vocabulary(TRAIN, max_size=4, min_df=1)
    for texts in (TRAIN, ["totally unseen words only"]):
        assert one_hot_vectorize(texts, vocab).shape == (len(texts), 4)
        assert bow_vectorize(texts, vocab).shape == (len(texts), 4)


def test_one_hot_is_binary_and_bow_counts():
    vocab = {"free": 0, "now": 1}
    assert one_hot_vectorize(["free free now"], vocab).tolist() == [[1.0, 1.0]]
    assert bow_vectorize(["free free now"], vocab).tolist() == [[2.0, 1.0]]


def test_tfidf_matches_formula():
    vocab = {"free": 0, "now": 1}
    train = ["free now", "free", "other", "other"]          # N=4, DF(free)=2, DF(now)=1
    idf = compute_idf(train, vocab)
    assert np.allclose(idf, [np.log(4 / 2), np.log(4 / 1)])
    x = tfidf_vectorize(["free free now"], vocab, idf)      # TF = 2/3 and 1/3
    assert np.allclose(x, [[2 / 3 * np.log(2), 1 / 3 * np.log(4)]])


def test_word_in_every_message_gets_zero_idf():
    vocab = {"the": 0}
    assert compute_idf(["the a", "the b"], vocab).tolist() == [0.0]
