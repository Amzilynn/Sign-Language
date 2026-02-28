"""
train.py
========
Trains the ST-GCN model on pre-processed ASL landmark data.

Usage (from project root, with venv active):
    python scripts/train.py
    python scripts/train.py --epochs 30 --num_classes 2000

Prerequisites:
    Run scripts/preprocess_dataset.py first to generate data/processed/.
"""

import os, sys, json, argparse, logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.stgcn import DummySTGCN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RealSignLanguageDataset(Dataset):
    """
    Loads pre-processed .npy landmark arrays from data/processed/.

    X.npy  shape: (N, 3, T, 21)  float32
    y.npy  shape: (N,)            int64
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(processed_dir: str, val_split: float = 0.1):
    X_path = os.path.join(processed_dir, 'X.npy')
    y_path = os.path.join(processed_dir, 'y.npy')

    if not os.path.exists(X_path):
        raise FileNotFoundError(
            f"Processed data not found at {X_path}.\n"
            "Run:  python scripts/preprocess_dataset.py"
        )

    X = np.load(X_path)  # (N, 3, T, 21)
    y = np.load(y_path)  # (N,)
    N = len(y)

    # Simple sequential train/val split
    split_idx = int(N * (1 - val_split))
    indices = np.random.permutation(N)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_ds = RealSignLanguageDataset(X[train_idx], y[train_idx])
    val_ds   = RealSignLanguageDataset(X[val_idx],   y[val_idx])

    log.info(f"Dataset loaded: {N} samples, {len(np.unique(y))} classes.")
    log.info(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(epochs: int, batch_size: int, lr: float,
          processed_dir: str, num_classes: int, save_path: str):

    train_ds, val_ds = load_data(processed_dir)

    # Override num_classes from actual data if not specified
    actual_classes = int(np.load(os.path.join(processed_dir, 'y.npy')).max()) + 1
    if num_classes == 0:
        num_classes = actual_classes
    log.info(f"Training ST-GCN with {num_classes} output classes.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    model = DummySTGCN(in_channels=3, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += y_batch.size(0)

        train_acc = 100 * correct / max(total, 1)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).argmax(dim=1)
                v_correct += (preds == y_batch).sum().item()
                v_total   += y_batch.size(0)

        val_acc = 100 * v_correct / max(v_total, 1)
        scheduler.step()

        log.info(f"Epoch [{epoch:3d}/{epochs}]  "
                 f"Loss: {total_loss/len(train_loader):.4f}  "
                 f"Train Acc: {train_acc:.1f}%  Val Acc: {val_acc:.1f}%")

        # ── Save best ──────────────────────────────────────────────────────
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes,
                'val_acc': val_acc,
            }, save_path)

    log.info(f"Training complete. Best val acc: {best_val_acc:.1f}%")
    log.info(f"Model saved to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ST-GCN on processed ASL dataset.')
    parser.add_argument('--epochs',        type=int,   default=30)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--processed_dir', type=str,   default='data/processed')
    parser.add_argument('--num_classes',   type=int,   default=0,
                        help='0 = infer automatically from dataset.')
    parser.add_argument('--save_path',     type=str,   default='models/stgcn_best.pth')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    train(args.epochs, args.batch_size, args.lr,
          args.processed_dir, args.num_classes, args.save_path)
