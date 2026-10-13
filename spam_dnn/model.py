"""A programmatically built deep dense network (Phase 2)."""
import torch.nn as nn


def build_deep_dnn(input_dim, n_hidden_layers=41, width=64, init="he"):
    """input_dim -> [Linear(width) + ReLU] x n_hidden_layers -> Linear(1).

    The last layer outputs a raw score (logit); BCEWithLogitsLoss applies the
    sigmoid. init="he" applies He (Kaiming) normal initialisation to every
    Linear layer (zero biases); init="default" keeps PyTorch's own default and
    is only meant for the initialisation comparison.
    """
    layers, in_features = [], input_dim
    for _ in range(n_hidden_layers):
        layers += [nn.Linear(in_features, width), nn.ReLU()]
        in_features = width
    layers.append(nn.Linear(in_features, 1))
    model = nn.Sequential(*layers)

    if init == "he":
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)
    elif init != "default":
        raise ValueError("init must be 'he' or 'default'")
    return model
