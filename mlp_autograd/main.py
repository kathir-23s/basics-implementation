# mlp_autograd/main.py
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from mlp_autograd.models.mlp import MLP
from mlp_autograd.tensor.tensor_scratch import TensorT


import os
os.environ.setdefault("OMP_NUM_THREADS", "8")          # tune: set to your CPU cores
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

# ------------------------------------------------------------------
# 1. generate 2-D toy dataset
# ------------------------------------------------------------------
X_np, y_np = make_moons(n_samples=500, noise=0.2, random_state=42)
y_np = y_np.astype(np.float64).reshape(1, -1)          # shape (1, 500)

# train / test split  (sklearn gives row-major; transpose to (features, m))
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_np, y_np.T, test_size=0.2, random_state=42
)
X_train_np, X_test_np = X_train_np.T, X_test_np.T      # (2, m)
y_train_np, y_test_np = y_train_np.T, y_test_np.T      # (1, m)

print(f"Train shapes : X {X_train_np.shape}  y {y_train_np.shape}")
print(f"Test  shapes : X {X_test_np.shape}  y {y_test_np.shape}")

# ------------------------------------------------------------------
# 2. convert numpy → TensorT
# ------------------------------------------------------------------
X_train = TensorT(X_train_np.tolist())
y_train = TensorT(y_train_np.tolist())
X_test  = TensorT(X_test_np.tolist())
y_test  = TensorT(y_test_np.tolist())

# ------------------------------------------------------------------
# 3. build & train multilayer perceptron
# ------------------------------------------------------------------
mlp = MLP(
    input_size=2,
    hidden_layers=[4, 4, 4, 4, 4],
    output_size=1,
    weight_initialization='he_normal',
    activation_func='relu',
    loss_function='binary_cross_entropy_loss',
    learning_rate=0.05
)

costs = mlp.train(X_train, y_train, epochs=4000, print_cost_every=250)

# ------------------------------------------------------------------
# 4. evaluate
# ------------------------------------------------------------------
pred = mlp.predict(X_test)               # TensorT (1, m)
pred_labels = np.array(pred.data)        # to numpy
y_true      = np.array(y_test.data)

accuracy = np.mean(pred_labels == y_true) * 100.0
print(f"Test Accuracy: {accuracy:.2f}%")
