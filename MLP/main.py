import numpy as np
import time
from ActivationFunctions import ActivationFunction
from WeightInitialization import WeightInitializer
from LossFunctions import LossFunction

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

        
        self._set_activation_functions()
        self._set_loss_function()     
        
        self.weights = []
        self.biases = []
        self._initialize_parameters()
        
        
        self.cache = {}
        self.grads = {}

        print("MLP initialized successfully.")
        print(f"  - Architecture: {self.layer_sizes}")
        print(f"  - Hidden Activation: {self.activation_func_name}")
        print(f"  - Weight Initialization: {self.weight_init_method}")
        print(f"  - Loss Function: {self.loss_func_name}")

    def _set_activation_functions(self):
        activations = {
            'relu': (ActivationFunction.relu, ActivationFunction.relu_derivative),
            'sigmoid': (ActivationFunction.sigmoid, ActivationFunction.sigmoid_derivative),
            'tanh': (ActivationFunction.tanh, ActivationFunction.tanh_derivative),
            'leaky_relu': (ActivationFunction.leaky_relu, ActivationFunction.leaky_relu_derivative),
            'softmax': (ActivationFunction.softmax, ActivationFunction.softmax_derivative),
            'elu': (ActivationFunction.elu, ActivationFunction.elu_derivative),
            'swish': (ActivationFunction.swish, ActivationFunction.swish_derivative),
            'softplus': (ActivationFunction.softplus, ActivationFunction.softplus_derivative)
        }
        if self.activation_func_name not in activations:
            raise ValueError("Unsupported activation function.")
        self.hidden_activation, self.hidden_activation_derivative = activations[self.activation_func_name]
        
        
        if self.loss_func_name == 'binary_cross_entropy_loss':
            self.output_activation = ActivationFunction.sigmoid
            self.output_activation_derivative = ActivationFunction.sigmoid_derivative
        elif self.loss_func_name == 'categorical_cross_entropy_loss':
            self.output_activation = ActivationFunction.softmax
            self.output_activation_derivative = ActivationFunction.softmax_derivative
        elif self.loss_func_name == 'mean_squared_error':
            self.output_activation = ActivationFunction.identity
            self.output_activation_derivative = ActivationFunction.identity_derivative
        elif self.loss_func_name == 'hinge_loss':
            self.output_activation = ActivationFunction.relu
            self.output_activation_derivative = ActivationFunction.relu_derivative
        elif self.loss_func_name == 'cross_entropy_loss':
            self.output_activation = ActivationFunction.softmax
            self.output_activation_derivative = ActivationFunction.softmax_derivative
        elif self.loss_func_name == 'mean_absolute_error':
            self.output_activation = ActivationFunction.identity
            self.output_activation_derivative = ActivationFunction.identity_derivative  

        else: 
            self.output_activation = ActivationFunction.sigmoid

    def _set_loss_function(self):
        losses = {
            'binary_cross_entropy_loss': (LossFunction.binary_cross_entropy_loss, LossFunction.binary_cross_entropy_gradient),
            'categorical_cross_entropy_loss': (LossFunction.categorical_cross_entropy_loss, None), # Derivative is special case
            'mean_squared_error': (LossFunction.mean_squared_error, LossFunction.mean_squared_error_gradient),
            'hinge_loss': (LossFunction.hinge_loss, LossFunction.hinge_loss_gradient),
            'cross_entropy_loss': (LossFunction.cross_entropy_loss, LossFunction.cross_entropy_gradient),
            'mean_absolute_error': (LossFunction.mean_absolute_error, LossFunction.mean_absolute_error_gradient)
            # Add more loss functions here as needed
        }
        if self.loss_func_name not in losses:
            raise ValueError("Unsupported loss function.")
        self.cost, self.cost_derivative = losses[self.loss_func_name]

    def _initialize_parameters(self):
        initializers = {
            'random_normal': WeightInitializer.random_normal,
            'random_uniform': WeightInitializer.random_uniform,
            'xavier_uniform': WeightInitializer.xavier_uniform,
            'xavier_normal': WeightInitializer.xavier_normal,
            'he_uniform': WeightInitializer.he_uniform,
            'he_normal': WeightInitializer.he_normal,
            'zero': WeightInitializer.zero,
            'constant': WeightInitializer.constant,
            'lecun_uniform': WeightInitializer.lecun_uniform,
            'lecun_normal': WeightInitializer.lecun_normal,
            # 'xavier': WeightInitializer.xavier,
            # 'he': WeightInitializer.he
        }
        if self.weight_init_method not in initializers:
            raise ValueError("Unsupported weight initialization.")
        initializer = initializers[self.weight_init_method]

        for i in range(1, self.num_layers):
            input_dim, output_dim = self.layer_sizes[i-1], self.layer_sizes[i]
            self.weights.append(initializer(input_dim, output_dim))
            self.biases.append(np.zeros((output_dim, 1)))

    def forward(self, X):
        self.cache = {'A0': X}
        A = X
        for l in range(1, self.num_layers):
            A_prev = A
            W, b = self.weights[l-1], self.biases[l-1]
            Z = np.dot(W, A_prev) + b
            
            activation = self.hidden_activation if l < self.num_layers - 1 else self.output_activation
            A = activation(Z)
            
            self.cache[f'A{l}'], self.cache[f'Z{l}'] = A, Z
        return A

    def backward(self, Y):
        m = Y.shape[1]
        L = self.num_layers - 1
        AL = self.cache[f'A{L}']

        if self.loss_func_name == 'categorical_cross_entropy_loss':
            dZ = AL - Y
        else:
            dAL = self.cost_derivative(Y, AL)
            dZ = dAL * self.output_activation_derivative(self.cache[f'Z{L}'])

        A_prev = self.cache[f'A{L-1}']
        self.grads[f'dW{L}'] = (1/m) * np.dot(dZ, A_prev.T)
        self.grads[f'db{L}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dAPrev = np.dot(self.weights[L-1].T, dZ)

        for l in reversed(range(1, L)):
            dZ = dAPrev * self.hidden_activation_derivative(self.cache[f'Z{l}'])
            A_prev = self.cache[f'A{l-1}']
            self.grads[f'dW{l}'] = (1/m) * np.dot(dZ, A_prev.T)
            self.grads[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            dAPrev = np.dot(self.weights[l-1].T, dZ)

    def update_parameters(self):
        for l in range(1, self.num_layers):
            self.weights[l-1] -= self.learning_rate * self.grads[f'dW{l}']
            self.biases[l-1] -= self.learning_rate * self.grads[f'db{l}']

    def train(self, X, Y, epochs, print_cost_every=100):
        print(f"\n--- Starting Training for {epochs} epochs ---")
        start_time = time.time()
        costs = []

        for i in range(epochs):
            AL = self.forward(X)
            self.backward(Y)
            self.update_parameters()
            
            if i % print_cost_every == 0 or i == epochs - 1:
                cost = self.cost(Y, AL)
                costs.append(cost)
                print(f"Epoch {i: >4} | Cost: {cost:.6f}")
        
        end_time = time.time()
        print(f"--- Training finished in {end_time - start_time:.2f} seconds ---")
        return costs

    def predict(self, X):
        AL = self.forward(X)
        if self.loss_func_name == 'binary_crossentropy':
            return (AL > 0.5).astype(int)
        elif self.loss_func_name == 'categorical_crossentropy':
            return np.argmax(AL, axis=0)
        return AL

if __name__ == '__main__':
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split

 
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    y = y.reshape(1, -1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y.T, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T

    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test data shape:     X={X_test.shape}, y={y_test.shape}")

    mlp = MLP(
        input_size=2, 
        hidden_layers=[3, 5, 4], 
        output_size=1,
        weight_initialization='random_normal',
        activation_func='tanh',
        loss_function= 'mean_squared_error', #'binary_cross_entropy_loss',
        learning_rate=0.05
    )

    costs = mlp.train(X_train, y_train, epochs=2500, print_cost_every=250)

    predictions = mlp.predict(X_test)
    accuracy = np.mean(predictions == y_test) * 100
    print(f"\nTest Accuracy: {accuracy:.3f}%")
