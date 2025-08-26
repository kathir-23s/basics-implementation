
import time
from core.ActivationFunctions import ActivationFunction
from core.WeightInitialization import WeightInitializer
from core.LossFunctions import LossFunction
from tensor.tensor_scratch import TensorT
import pickle

class MLP:
    def __init__(self, input_size, hidden_layers, output_size, 
                 weight_initialization='he_normal', activation_func='relu', 
                 loss_function='binary_cross_entropy_loss', learning_rate=0.01):

        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        self.num_layers = len(self.layer_sizes)
        self.learning_rate = learning_rate
        self.weight_init_method = weight_initialization
        self.activation_func_name = activation_func
        self.loss_func_name = loss_function

        # Resolve everything from the libraries
        self.hidden_activation, self.hidden_activation_derivative = ActivationFunction.get(self.activation_func_name)
        self.output_activation, self.output_activation_derivative = ActivationFunction.output_for_loss(self.loss_func_name)
        self.cost, self.cost_derivative = LossFunction.get(self.loss_func_name)
        self.initializer = WeightInitializer.get(self.weight_init_method)

        self.weights = []
        self.biases = []
        self._initialize_parameters()

        print("MLP initialized successfully.")
        print(f"  - Architecture: {self.layer_sizes}")
        print(f"  - Hidden Activation: {self.activation_func_name}")
        print(f"  - Output Activation (from loss): {self.output_activation.__name__}")
        print(f"  - Weight Initialization: {self.weight_init_method}")
        print(f"  - Loss Function: {self.loss_func_name}")

    def _initialize_parameters(self):
        
        for i in range(1, self.num_layers):
            in_dim, out_dim = self.layer_sizes[i-1], self.layer_sizes[i]
            # Expect initializer(in_dim, out_dim) -> W of shape (out_dim, in_dim)
            W = self.initializer(in_dim, out_dim)
            b = TensorT.unit_tensor(0.0, (out_dim, 1)) 
             # Bias shape (out_dim, 1)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        A = X

        for l in range(1, self.num_layers):
            W, b = self.weights[l-1], self.biases[l-1]
            Z = W.tmatmul_fast(A)
            Z = Z + b
            act = self.hidden_activation if l < self.num_layers - 1 else self.output_activation
            A = act(Z)

            if l == self.num_layers - 1:
                self._last_logits = Z       # TensorT before softmax
                self._last_probs  = A       # TensorT after softmax
        return A

    def zero_grad(self):
        """Zero out gradients for all parameters"""
        for w in self.weights:
            w.grad = None
        for b in self.biases:
            b.grad = None                                  

    def update_parameters(self):
        for w in self.weights:
            if w.grad is not None:
                grad_tensor = TensorT(w.grad)   # wrap grad as TensorT
                update = grad_tensor * self.learning_rate   # scale by lr
                w_new = w - update
                w.data = w_new.data  # Update the weight data

        for b in self.biases:
            if b.grad is not None:
                grad_tensor = TensorT(b.grad)   # wrap grad as TensorT
                update = grad_tensor * self.learning_rate   # scale by lr
                b_new = b - update
                b.data = b_new.data


    # def train(self, X, Y, epochs, print_cost_every=100):
    #     print(f"\n--- Starting Training for {epochs} epochs ---")
    #     start_time = time.time()
    #     costs = []

    #     for i in range(epochs):
    #         self.zero_grad()  # Reset gradients
    #         AL = self.forward(X)
    #         loss = self.cost(Y, AL)
    #         loss.zero_grad()  # Reset gradients for loss
    #         loss.backward()  # Compute gradients
    #         self.update_parameters()  # Update weights and biases

    #         if i % print_cost_every == 0 or i == epochs - 1:
    #         # Extract scalar loss value
    #             cost = loss.data[0][0] if hasattr(loss, 'data') else float(loss)
    #             costs.append(cost)
    #             print(f"Epoch {i:>4} | Cost: {cost:.6f}")
            
    #     dt = time.time() - start_time
    #     print(f"--- Training Finished in {dt:.2f} seconds ---")
    #     return costs

    def _batch_slices(self, N, batch_size):
        s = 0
        while s < N:
            e = min(s + batch_size, N)
            yield s, e
            s = e

    def train(self, X, Y, epochs, print_cost_every=100, batch_size=128, shuffle=True):
        """
        Mini-batch SGD. X: (d, N), Y: (C, N)
        """
        import time, random
        print(f"\n--- Starting Training for {epochs} epochs (batch_size={batch_size}) ---")
        start_time = time.time()
        costs = []
        N = X.shape[1]

        # 1-time index list for shuffling columns
        idx = list(range(N))

        for ep in range(epochs):
            if shuffle:
                random.shuffle(idx)

            # views as simple column-reordered lists (cheap, pure Python)
            X_cols = [[X.data[i][j] for j in idx] for i in range(X.shape[0])]
            Y_cols = [[Y.data[i][j] for j in idx] for i in range(Y.shape[0])]

            epoch_loss = 0.0
            batches = 0

            for s, e in self._batch_slices(N, batch_size):
                # Build batch tensors (shape: (d, B), (C, B))
                Xb = TensorT([row[s:e] for row in X_cols])
                Yb = TensorT([row[s:e] for row in Y_cols])

                self.zero_grad()
                AL = self.forward(Xb)
                loss = self.cost(Yb, AL)     # your CE should average over B
                loss.backward()
                self.update_parameters()

                epoch_loss += loss.data[0][0]
                batches += 1

            if ep % print_cost_every == 0 or ep == epochs - 1:
                avg_loss = epoch_loss / max(1, batches)
                costs.append(avg_loss)
                print(f"Epoch {ep:>4} | Avg loss: {avg_loss:.6f}")

        dt = time.time() - start_time
        print(f"--- Training Finished in {dt:.2f} seconds ---")
        return costs

    def predict(self, X):
        AL = self.forward(X)
        
        # Binary case → threshold at 0.5
        if self.loss_func_name == 'binary_cross_entropy_loss' and self.layer_sizes[-1] == 1:
            # Manual thresholding for TensorT
            threshold_data = AL._apply_unary(AL.data, lambda x: 1.0 if x > 0.5 else 0.0)
            return TensorT(threshold_data)
        
        # Multiclass → argmax along rows (for each sample)
        if self.loss_func_name in ('categorical_cross_entropy_loss', 'cross_entropy_loss'):
            # Manual argmax for TensorT (column-wise - each column is a sample)
            # predictions = []
            # for col_idx in range(AL.shape[1]):  # For each sample
            #     column = [AL.data[row][col_idx] for row in range(AL.shape[0])]
            #     max_idx = column.index(max(column))
            #     predictions.append(max_idx)
            
            # # Return as row vector: (1, num_samples)
            # return TensorT([predictions])
            return AL
        
        # Fallback: return raw activations
        return AL

        
    def save_weights(self, filepath="mlp_weights.pkl"):
        weights_biases = {
            "weights": [w.data for w in self.weights],
            "biases": [b.data for b in self.biases]
        }
        with open(filepath, "wb") as f:
            pickle.dump(weights_biases, f)
        print(f"[DEBUG] Weights saved to {filepath}")

    def load_weights(self, filepath="mlp_weights.pkl"):
        with open(filepath, "rb") as f:
            weights_biases = pickle.load(f)
        for w, w_data in zip(self.weights, weights_biases["weights"]):
            w.data = w_data
        for b, b_data in zip(self.biases, weights_biases["biases"]):
            b.data = b_data
        print(f"[DEBUG] Weights loaded from {filepath}")

    # mlp.py
    def debug_check_last_grad(self, Y):
        """
        Call after a forward+loss.backward() pass.
        Compares dZ_L (stored on self._last_logits.grad) vs (S - Y)/B.
        """
        Z = self._last_logits            # TensorT
        S = self._last_probs             # TensorT
        B = S.shape[1]

        # Expected: (S - Y) / B  -> plain list
        exp = [[(S.data[i][j] - Y.data[i][j]) / B
                for j in range(B)]
            for i in range(S.shape[0])]

        got = Z.grad                     # what backprop produced
        # compute max |diff|
        max_diff = 0.0
        for i in range(len(exp)):
            for j in range(len(exp[0])):
                d = abs(exp[i][j] - got[i][j])
                if d > max_diff:
                    max_diff = d
        print(f"[DEBUG] dZ_L check: max |(S-Y)/B - dZ| = {max_diff:.3e}")

