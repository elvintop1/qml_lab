# qml_lab/baselines/deeplearning.py
from __future__ import annotations

import numpy as np

def has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def train_mlp_tabular(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
    *, hidden: int = 128, epochs: int = 10, batch_size: int = 128, lr: float = 1e-3,
    seed: int = 0, verbose: bool = False
):

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as e:
        raise RuntimeError("PyTorch not available, cannot run MLP baseline") from e

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32)
    in_dim = int(Xtr.shape[1])
    n_classes = int(np.unique(ytr).size)

    class MLP(nn.Module):
        def __init__(self, in_dim: int, n_classes: int):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.fc2 = nn.Linear(hidden, n_classes)
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for ep in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        if verbose:
            print(f"[MLP] epoch {ep+1}/{epochs} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(Xte_t.to(device))
        proba = torch.softmax(logits, dim=1).cpu().numpy()

    return proba, "MLP (tabular)"


def train_cnn_images(
    Xtr_img: np.ndarray, ytr: np.ndarray, Xte_img: np.ndarray,
    *, epochs: int = 5, batch_size: int = 128, lr: float = 1e-3,
    seed: int = 0, verbose: bool = False
):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as e:
        raise RuntimeError("PyTorch not available, cannot run CNN baseline") from e

    torch.manual_seed(seed)

    if Xtr_img.ndim == 3:
        Xtr_img = Xtr_img[:, None, :, :]
    if Xte_img.ndim == 3:
        Xte_img = Xte_img[:, None, :, :]

    Xtr_img = Xtr_img.astype(np.float32, copy=False)
    Xte_img = Xte_img.astype(np.float32, copy=False)

    n_classes = int(np.unique(ytr).size)
    Ntr, C, H, W = Xtr_img.shape

    class SmallCNN(nn.Module):
        def __init__(self, C: int, H: int, W: int, n_classes: int):
            super().__init__()
            self.conv1 = nn.Conv2d(C, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2)
            h = H // 4
            w = W // 4
            self.fc1 = nn.Linear(64 * h * w, 128)
            self.fc2 = nn.Linear(128, n_classes)
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(C, H, W, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()

    Xtr_t = torch.tensor(Xtr_img, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = torch.tensor(Xte_img, dtype=torch.float32)

    dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for ep in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
        if verbose:
            print(f"[CNN] epoch {ep+1}/{epochs} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(Xte_t.to(device))
        proba = torch.softmax(logits, dim=1).cpu().numpy()

    return proba, "CNN (2conv)"
