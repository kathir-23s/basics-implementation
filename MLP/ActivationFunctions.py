import numpy as np
class ActivationFunction:

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        s = ActivationFunction.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        dZ = np.array(z, copy=True)
        dZ[dZ <= 0] = 0
        dZ[dZ > 0] = 1
        return dZ
        
    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        return 1 - np.power(ActivationFunction.tanh(z), 2)

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    @staticmethod
    def softmax_derivative(z):  
        s = ActivationFunction.softmax(z)
        return s * (1 - s)

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)

    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        dz = np.ones_like(z)
        dz[z < 0] = alpha
        return dz

    @staticmethod
    def elu(z, alpha=1.0):
        return np.where(z >= 0, z, alpha * (np.exp(z) - 1))

    @staticmethod
    def elu_derivative(z, alpha=1.0):
        return np.where(z >= 0, 1, alpha * np.exp(z))

    @staticmethod
    def swish(z, beta=1.0):
        return z * ActivationFunction.sigmoid(beta * z)

    @staticmethod
    def swish_derivative(z, beta=1.0):
        sig = ActivationFunction.sigmoid(beta * z)
        return sig + beta * z * sig * (1 - sig)

    @staticmethod
    def softplus(z):
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)

    @staticmethod
    def softplus_derivative(z):
        return ActivationFunction.sigmoid(z)
    
    @staticmethod
    def identity(z):
        return z

    @staticmethod
    def identity_derivative(z):
        return np.ones_like(z)
     