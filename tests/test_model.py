import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from spam_dnn.model import build_deep_dnn  # noqa: E402


def test_more_than_40_hidden_layers_and_output_shape():
    model = build_deep_dnn(input_dim=500, n_hidden_layers=41, width=64)
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 42                       # 41 hidden + 1 output
    assert linears[0].in_features == 500            # input matches the vector size
    assert model(torch.randn(8, 500)).shape == (8, 1)


def test_he_initialisation_scale_and_zero_bias():
    torch.manual_seed(0)
    model = build_deep_dnn(input_dim=500, n_hidden_layers=3, width=256, init="he")
    hidden = [m for m in model.modules() if isinstance(m, nn.Linear)][1]
    expected_std = (2.0 / hidden.in_features) ** 0.5
    assert abs(hidden.weight.std().item() - expected_std) / expected_std < 0.05
    assert torch.count_nonzero(hidden.bias) == 0
