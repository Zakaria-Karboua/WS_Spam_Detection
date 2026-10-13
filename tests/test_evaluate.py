from spam_dnn.evaluate import compute_metrics


def test_metrics_on_known_confusion_matrix():
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]       # TP=3 FN=1 FP=1 TN=5
    m = compute_metrics(y_true, y_pred)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 5)
    assert m["accuracy"] == 0.8
    assert m["precision"] == 0.75
    assert m["recall"] == 0.75
    assert m["f1"] == 0.75
    assert abs(m["fpr"] - 1 / 6) < 1e-12


def test_no_division_by_zero():
    m = compute_metrics([0, 0], [0, 0])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
