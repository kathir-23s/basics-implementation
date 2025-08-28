import os
os.environ["OMP_NUM_THREADS"] = "8"   # set threads before OpenMP runtime starts

from mm_backend import set_backend, C_BACKEND
set_backend(C_BACKEND)


import time, random, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
# from tensorflow.keras.datasets import mnist
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from matmul_c_tensorT import TensorT
from models.mlp import MLP


# os.environ["OMP_NUM_THREADS"] = "8"




# -----------------------------
# 1) Data loading & preprocessing
# -----------------------------
# Priority 1: local npz at ./data/mnist.npz
local_npz = "data/mnist.npz"
if os.path.exists(local_npz):
    with np.load(local_npz) as d:
        X_train_np = d["x_train"]
        y_train_np = d["y_train"]
        X_test_np  = d["x_test"]
        y_test_np  = d["y_test"]
else:
    # Priority 2: fetch via OpenML (needs internet)
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X_all = mnist.data.reshape(-1, 28, 28).astype(np.uint8)
        y_all = mnist.target.astype(np.int64)
        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            X_all, y_all, test_size=10000, random_state=42, stratify=y_all
        )
    except Exception as e:
        raise RuntimeError(
            "No local MNIST and OpenML fetch failed. "
            "Place mnist.npz at ./data/mnist.npz or ensure internet."
        ) from e

print(f"[DEBUG] Raw MNIST shapes: train {X_train_np.shape}, test {X_test_np.shape}")

# Flatten to vectors and normalize
X_train_flat = X_train_np.reshape(X_train_np.shape[0], -1) / 255.0
X_test_flat  = X_test_np.reshape(X_test_np.shape[0], -1) / 255.0

# Split train into train/val
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X_train_flat, y_train_np, test_size=0.1, random_state=42
)

# One-hot encode labels
# One-hot encode labels
def one_hot(y, num_classes=10):
    oh = np.zeros((y.size, num_classes))
    oh[np.arange(y.size), y] = 1
    return oh

Y_train_np = one_hot(y_train_np)
Y_val_np   = one_hot(y_val_np)
Y_test_np  = one_hot(y_test_np)   # <-- fixed: use y_test_np, not y_test_raw

# Convert to TensorT (shape: (features, samples))
X_train = TensorT(X_train_np.T.tolist())
Y_train = TensorT(Y_train_np.T.tolist())
X_val_t = TensorT(X_val_np.T.tolist())
Y_val_t = TensorT(Y_val_np.T.tolist())
X_test  = TensorT(X_test_flat.T.tolist())

# Keep raw test labels for evaluation
y_test_raw = y_test_np           # rename for consistency with rest of code
   # keep raw labels for final eval

print("[DEBUG] Tensor shapes:",
      f"X_train {X_train.shape}, Y_train {Y_train.shape},",
      f"X_val {X_val_t.shape}, Y_val {Y_val_t.shape},",
      f"X_test {X_test.shape}")


# -----------------------------
# 2) MLP init
# -----------------------------
mlp = MLP(
    input_size=784,
    hidden_layers=[128, 64],
    output_size=10,
    weight_initialization="he_normal",
    activation_func="relu",
    loss_function="cross_entropy_loss",
    learning_rate=0.01
)

# -----------------------------
# 3) Training loops
# -----------------------------
def train_debug(self, X, Y, X_val=None, Y_val=None, epochs=10, print_every=1):
    print(f"\n--- Starting MNIST Training for {epochs} epochs ---")
    t0 = time.time()
    train_hist = []

    for ep in range(1, epochs+1):
        # Forward + loss
        AL = self.forward(X)
        loss = self.cost(Y, AL)

        # Backprop
        loss.zero_grad()
        loss.backward()

        # DEBUG: last-layer gradient sanity (only first 3 epochs)
        if ep <= 3 and hasattr(self, 'debug_check_last_grad'):
            self.debug_check_last_grad(Y)

        # DEBUG: mean |grad| per layer BEFORE update
        for l, (W, B) in enumerate(zip(self.weights, self.biases), start=1):
            gW = 0.0; nW = 0
            if W.grad:
                for row in W.grad:
                    for v in row:
                        gW += abs(v); nW += 1
            gB = 0.0; nB = 0
            if B.grad:
                for row in B.grad:
                    for v in row:
                        gB += abs(v); nB += 1
            # print(f"[grad] L{l}  |W|_mean={gW/max(1,nW):.3e}  |b|_mean={gB/max(1,nB):.3e}")

        # stash params, then update
        prev_W = [[row[:] for row in W.data] for W in self.weights]
        prev_b = [[row[:] for row in b.data] for b in self.biases]

        self.update_parameters()

        # DEBUG: how much params changed
        for l, (W, B) in enumerate(zip(self.weights, self.biases), start=1):
            dW = 0.0
            for i in range(len(W.data)):
                for j in range(len(W.data[0])):
                    dW += abs(W.data[i][j] - prev_W[l-1][i][j])
            dB = 0.0
            for i in range(len(B.data)):
                dB += abs(B.data[i][0] - prev_b[l-1][i][0])
            # print(f"[step] L{l}  ΔW_sum={dW:.3e}  Δb_sum={dB:.3e}")

        # Log once per epoch
        if (ep % print_every) == 0 or ep == epochs:
            tr_loss = float(loss.data[0][0])
            msg = f"Epoch {ep:>3} | train_loss: {tr_loss:.4f}"
            if X_val is not None and Y_val is not None:
                logits = self.forward(X_val)
                pred_val = np.argmax(np.array(logits.data), axis=0)
                true_val = np.argmax(np.array(Y_val.data), axis=0)
                val_acc = (pred_val == true_val).mean()
                msg += f" | val_acc: {val_acc*100:.2f}%"
            print(msg)
            train_hist.append(tr_loss)

    print(f"--- Finished in {time.time()-t0:.2f}s ---")
    return train_hist


def train_minibatch_debug(self, X, Y, X_val=None, Y_val=None, epochs=10, print_every=1,
                          batch_size=128, shuffle=True):
    print(f"\n--- Starting MNIST Training for {epochs} epochs (batch_size={batch_size}) ---")
    t0 = time.time()
    result = []

    N, D, C = X.shape[1], X.shape[0], Y.shape[0]

    for ep in range(1, epochs+1):
        idx = list(range(N))
        if shuffle:
            random.shuffle(idx)

        epoch_loss_weighted = 0.0
        seen = 0
        first_batch_done = False

        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            cols = idx[s:e]
            B = e - s

            Xb = TensorT([[X.data[i][j] for j in cols] for i in range(D)])
            Yb = TensorT([[Y.data[i][j] for j in cols] for i in range(C)])

            AL = self.forward(Xb)
            loss = self.cost(Yb, AL)
            loss.zero_grad()
            loss.backward()

            if not first_batch_done:
                if ep <= 3 and hasattr(self, 'debug_check_last_grad'):
                    self.debug_check_last_grad(Yb)
                first_batch_done = True

            self.update_parameters()

            batch_loss = float(loss.data[0][0])
            epoch_loss_weighted += batch_loss * B
            seen += B

        if (ep % print_every) == 0 or ep == epochs:
            avg_loss = epoch_loss_weighted / max(1, seen)
            msg = f"Epoch {ep:>3} | train_loss: {avg_loss:.4f}"
            if X_val is not None and Y_val is not None:
                logits = self.forward(X_val)
                pred_val = np.argmax(np.array(logits.data), axis=0)
                true_val = np.argmax(np.array(Y_val.data), axis=0)
                val_acc = (pred_val == true_val).mean() * 100.0
                msg += f" | val_acc: {val_acc:.2f}%"
            print(msg)
            result.append(avg_loss)

    print(f"--- Finished in {time.time()-t0:.2f}s ---")
    return result


# -----------------------------
# 4) Plotting utilities
# -----------------------------
def show_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def preview_samples(X_flat, y_true, y_pred, count=10, save_path="predictions_preview.png"):
    idx = np.random.choice(X_flat.shape[0], count, replace=False)
    fig, axes = plt.subplots(2, count, figsize=(1.2*count, 2.6))
    for k, ax in enumerate(axes[0]):
        img = X_flat[idx[k]].reshape(28, 28)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"T:{y_true[idx[k]]}")
    for k, ax in enumerate(axes[1]):
        ax.axis("off")
        ax.set_title(f"P:{y_pred[idx[k]]}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 5) Main entry point
# -----------------------------
def run_train(
    use_minibatch: bool = True,
    epochs: int = 10,
    batch_size: int = 128,
    shuffle: bool = True,
):
    RUN_TRAIN = True
    RUN_SMOKE = False

    if RUN_TRAIN:
        if use_minibatch:
            costs = train_minibatch_debug(
                mlp, X_train, Y_train, X_val_t, Y_val_t,
                epochs=epochs, print_every=1,
                batch_size=batch_size, shuffle=shuffle
            )
        else:
            costs = train_debug(
                mlp, X_train, Y_train, X_val_t, Y_val_t,
                epochs=epochs, print_every=1
            )

        # Save & evaluate
        mlp.save_weights("mnist_mlp_weights.pkl")
        logits_test = mlp.predict(X_test)
        probs = np.array(logits_test.data)
        if probs.shape[0] != 10 and probs.shape[1] == 10:
            probs = probs.T
        y_pred = np.argmax(probs, axis=0)
        y_true = y_test_np
        acc = (y_pred == y_true).mean()
        print(f"\n=== Test Metrics ===")
        print(f"Accuracy: {acc*100:.2f}%")
        print(classification_report(y_true, y_pred, digits=4))
        show_confusion_matrix(y_true, y_pred,
                              title="MNIST Confusion Matrix",
                              save_path="confusion_matrix.png")
        preview_samples(X_test_flat, y_true, y_pred,
                        count=10,
                        save_path="predictions_preview.png")
        print("Saved plots: confusion_matrix.png, predictions_preview.png")

    if RUN_SMOKE:
        mlp2 = MLP(
            input_size=784,
            hidden_layers=[128, 64],
            output_size=10,
            weight_initialization="he_normal",
            activation_func="relu",
            loss_function="cross_entropy_loss",
            learning_rate=0.01
        )
        mlp2.load_weights("mnist_mlp_weights_10.pkl")
        logits = mlp2.forward(X_test)
        y_pred = np.argmax(np.array(logits.data), axis=0)
        y_true = y_test_np
        acc = (y_pred == y_true).mean() * 100.0
        print(f"[SMOKE] Reloaded weights OK — test accuracy: {acc:.2f}%")

def main():
    # Default run: minibatch, 10 epochs, bs=128
    run_train(use_minibatch=False, epochs=10, batch_size=128, shuffle=True)

if __name__ == "__main__":
    main()
