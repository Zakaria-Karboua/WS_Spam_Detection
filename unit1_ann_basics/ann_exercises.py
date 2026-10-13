import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# ---------------------------------------------------------------
# Shared setup (same data and model as ann_pytorch.py)
# ---------------------------------------------------------------
torch.manual_seed(42)

X = torch.randn(1000, 20)
y = (X[:, 0] + X[:, 1] > 0).float()
X = (X - X.mean(0)) / X.std(0)

ds = TensorDataset(X, y)
train, val, test = random_split(
    ds, [700, 150, 150], generator=torch.Generator().manual_seed(42)
)
train_dl = DataLoader(train, batch_size=64, shuffle=True)
val_dl = DataLoader(val, batch_size=64)
test_dl = DataLoader(test, batch_size=64)


class ANN(nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.fc1 = nn.Linear(inp, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


def accuracy(model, dl):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in dl:
            pred = (torch.sigmoid(model(xb)) > 0.5).float()
            correct += (pred == yb).sum().item()
            total += len(yb)
    return correct / total


def train_model(model, epochs=30, use_zero_grad=True, dl=train_dl):
    """Trains the model and returns the average loss of the last epoch."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_loss = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in dl:
            if use_zero_grad:
                optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
        epoch_loss /= len(dl.dataset)
    return epoch_loss


# ---------------------------------------------------------------
# Step 1 - Create a tensor of shape (4, 2) and print its mean
# ---------------------------------------------------------------
print("\n--- Step 1 ---")
t = torch.randn(4, 2)
print(t)
print("mean:", t.mean())

# ---------------------------------------------------------------
# Step 2 - Change batch_size to 16 and check len(train_dl)
# ---------------------------------------------------------------
print("\n--- Step 2 ---")
train_dl_16 = DataLoader(train, batch_size=16, shuffle=True)
print("batch_size=64 ->", len(train_dl), "batches")      # 11
print("batch_size=16 ->", len(train_dl_16), "batches")   # 44 (700 / 16 = 43.75, rounded up)

# ---------------------------------------------------------------
# Step 3 - Add a second hidden layer of size 16, check output shape
# ---------------------------------------------------------------
print("\n--- Step 3 ---")
class ANN2(nn.Module):
    def __init__(self, inp, h1, h2, out):
        super().__init__()
        self.fc1 = nn.Linear(inp, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


model2 = ANN2(20, 32, 16, 1)
print("output shape:", model2(X[:4]).shape)  # torch.Size([4]) - same as before

# ---------------------------------------------------------------
# Step 4 - What is CrossEntropyLoss used for?
# Multi-class classification (3+ classes). It takes raw scores
# (logits) of shape (N, C) and integer class labels of shape (N,).
# BCEWithLogitsLoss is for yes/no (2 classes) problems.
# ---------------------------------------------------------------
print("\n--- Step 4 ---")
logits = torch.randn(4, 3)               # 4 samples, 3 classes
targets = torch.tensor([0, 2, 1, 2])     # class index for each sample
ce = nn.CrossEntropyLoss()
print("cross-entropy loss:", ce(logits, targets).item())

# ---------------------------------------------------------------
# Step 5 - Remove zero_grad() and see what happens to the loss
# Without it, gradients pile up across batches, so each update
# uses the sum of all previous gradients instead of just the
# current batch.
# ---------------------------------------------------------------
print("\n--- Step 5 ---")
torch.manual_seed(0)
m_with = ANN(20, 32, 1)
loss_with = train_model(m_with, epochs=30, use_zero_grad=True)

torch.manual_seed(0)
m_without = ANN(20, 32, 1)
loss_without = train_model(m_without, epochs=30, use_zero_grad=False)

print(f"final loss WITH zero_grad:    {loss_with:.4f}")
print(f"final loss WITHOUT zero_grad: {loss_without:.4f}")

# ---------------------------------------------------------------
# Step 6 - Train for 5 epochs, then 100, compare validation accuracy
# ---------------------------------------------------------------
print("\n--- Step 6 ---")
for n_epochs in (5, 100):
    torch.manual_seed(0)
    m = ANN(20, 32, 1)
    train_model(m, epochs=n_epochs)
    print(
        f"{n_epochs:>3} epochs -> train acc {accuracy(m, train_dl):.3f} | "
        f"val acc {accuracy(m, val_dl):.3f}"
    )

# ---------------------------------------------------------------
# Step 7 - Check new_model gives the same test accuracy as the original
# ---------------------------------------------------------------
print("\n--- Step 7 ---")
torch.manual_seed(0)
original = ANN(20, 32, 1)
train_model(original, epochs=30)
torch.save(original.state_dict(), "ann_exercise.pt")

new_model = ANN(20, 32, 1)
new_model.load_state_dict(torch.load("ann_exercise.pt"))
new_model.eval()

print("original test acc:", accuracy(original, test_dl))
print("reloaded test acc:", accuracy(new_model, test_dl))
