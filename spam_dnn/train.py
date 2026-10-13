"""End-to-end training with Binary Cross-Entropy (Phase 3)."""
import math

import torch
import torch.nn as nn


def train_model(model, X_train, y_train, epochs=1000, lr=1e-4,
                grad_epochs=None, log_every=100, verbose=True, device=None):
    """Full-batch training with BCEWithLogitsLoss + Adam.

    Returns a history dict:
      loss, train_acc : one value per epoch
      grad_norms      : {epoch: [gradient norm of each Linear layer, input -> output]}
                        recorded at `grad_epochs` (default: first, middle, last epoch)
      diverged        : True if the loss became NaN/inf (training stops there)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    grad_epochs = set(grad_epochs) if grad_epochs else {1, max(1, epochs // 2), epochs}

    history = {"loss": [], "train_acc": [], "grad_norms": {}, "diverged": False}
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X).squeeze(-1)
        loss = criterion(logits, y)
        loss.backward()
        if epoch in grad_epochs:
            history["grad_norms"][epoch] = [l.weight.grad.norm().item() for l in linears]
        optimizer.step()

        loss_value = loss.item()
        acc = ((logits.detach() > 0).float() == y).float().mean().item()
        history["loss"].append(loss_value)
        history["train_acc"].append(acc)

        if not math.isfinite(loss_value):
            history["diverged"] = True
            if verbose:
                print(f"  epoch {epoch}: loss is {loss_value}, stopping")
            break
        if verbose and log_every and (epoch % log_every == 0 or epoch == 1):
            print(f"  epoch {epoch:>4} | loss {loss_value:.4f} | train acc {acc:.2%}")
    return history


def predict(model, X):
    """0/1 predictions (logit > 0 is the same as probability > 0.5)."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32, device=device)).squeeze(-1)
    return (logits > 0).int().cpu().numpy()
