
import time
from core.ActivationFunctions import ActivationFunction
from core.WeightInitialization import WeightInitializer
from core.LossFunctions import LossFunction
from tensor.tensor_scratch import TensorT

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
        # print(f"Initial A type: {type(A)}")
        # print(f"Initial A class: {A.__class__}")
        
        for l in range(1, self.num_layers):
            W, b = self.weights[l-1], self.biases[l-1]

            Z = W.block_matmul(A, 32) + b
            
            act = self.hidden_activation if l < self.num_layers - 1 else self.output_activation
            A = act(Z)
        
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


    def train(self, X, Y, epochs, print_cost_every=100):
        print(f"\n--- Starting Training for {epochs} epochs ---")
        start_time = time.time()
        costs = []

        for i in range(epochs):
            self.zero_grad()  # Reset gradients

            AL = self.forward(X)

            loss = self.cost(Y, AL)

            loss.zero_grad()  # Reset gradients for loss
            loss.backward()  # Compute gradients

            self.update_parameters()  # Update weights and biases

            if i % print_cost_every == 0 or i == epochs - 1:
            # Extract scalar loss value
                cost = loss.data[0][0] if hasattr(loss, 'data') else float(loss)
                costs.append(cost)
                print(f"Epoch {i:>4} | Cost: {cost:.6f}")
            
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
            predictions = []
            for col_idx in range(AL.shape[1]):  # For each sample
                column = [AL.data[row][col_idx] for row in range(AL.shape[0])]
                max_idx = column.index(max(column))
                predictions.append(max_idx)
            
            # Return as row vector: (1, num_samples)
            return TensorT([predictions])
        
        # Fallback: return raw activations
        return AL
