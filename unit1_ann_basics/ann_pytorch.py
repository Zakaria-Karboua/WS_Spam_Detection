# pip install torch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# ---------------------------------------------------------------
# 1. Set up and understand tensors
# ---------------------------------------------------------------
torch.manual_seed(42)

x = torch.randn(5, 3)  # 5 samples, 3 features
print(x.shape)         # torch.Size([5, 3])

# Try it: create a tensor of shape (4, 2) and print its mean
t = torch.randn(4, 2)
print(t.mean())

# ---------------------------------------------------------------
# 2. Prepare the data
# ---------------------------------------------------------------
X = torch.randn(1000, 20)
y = (X[:, 0] + X[:, 1] > 0).float()
X = (X - X.mean(0)) / X.std(0)

ds = TensorDataset(X, y)
train, val, test = random_split(ds, [700, 150, 150])

train_dl = DataLoader(train, batch_size=64, shuffle=True)
val_dl = DataLoader(val, batch_size=64)
test_dl = DataLoader(test, batch_size=64)

# Try it: change batch_size to 16 and check len(train_dl)

# ---------------------------------------------------------------
# 3. Define the model
# ---------------------------------------------------------------
class ANN(nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.fc1 = nn.Linear(inp, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


model = ANN(20, 32, 1)
print(model(X[:4]))  # 4 raw scores

# Try it: add a second hidden layer of size 16 and check the output shape

# ---------------------------------------------------------------
# 4. Choose a loss and an optimizer
# ---------------------------------------------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------------------
# 5. Write the training loop
# ---------------------------------------------------------------
for epoch in range(30):
    model.train()
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}, loss {loss.item():.4f}")

# Try it: remove zero_grad() and see what happens to the loss, then put it back

# ---------------------------------------------------------------
# 6. Evaluate on validation and test data
# ---------------------------------------------------------------
def accuracy(model, dl):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in dl:
            pred = (torch.sigmoid(model(xb)) > 0.5).float()
            correct += (pred == yb).sum().item()
            total += len(yb)
    return correct / total


print("val", accuracy(model, val_dl))
print("test", accuracy(model, test_dl))

# Try it: train for 5 epochs, then 100, and compare validation accuracy

# ---------------------------------------------------------------
# 7. Save the model and predict
# ---------------------------------------------------------------
torch.save(model.state_dict(), "ann.pt")

new_model = ANN(20, 32, 1)
new_model.load_state_dict(torch.load("ann.pt"))
new_model.eval()

sample = torch.randn(1, 20)
prob = torch.sigmoid(new_model(sample))
print(prob.item())

# Try it: check that new_model gives the same test accuracy as the original
print("test (reloaded)", accuracy(new_model, test_dl))
