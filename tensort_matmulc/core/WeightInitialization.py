import random
import math
from matmul_c_tensorT import TensorT

class WeightInitializer:

    @staticmethod
    def zero(input_dim, output_dim):
        """
        Initializes all weights to zero.
        """
        # return np.zeros((output_dim, input_dim))
        return TensorT.unit_tensor(0, (output_dim, input_dim))

    @staticmethod
    def ones(input_dim, output_dim, value=0.01):
        """
        Initializes all weights to a constant value.
        """
        # return np.full((output_dim, input_dim), value)
        return TensorT.const_tensor(value, (output_dim, input_dim))

    
    @staticmethod
    def random_uniform(input_dim, output_dim, low=-0.05, high=0.05):
        """
        Initializes weights with a uniform distribution.
        """
        # return np.random.uniform(low, high, (output_dim, input_dim))

        data = [[random.uniform(low, high) for _ in range(input_dim)] for _ in range(output_dim)]
        return TensorT(data)

    @staticmethod
    def random_normal(input_dim, output_dim, mean=0.0, std=0.05):
        """
        Initializes weights with a normal distribution.
        """
        # return np.random.normal(mean, std, (output_dim, input_dim))
        data = [[random.gauss(mean, std) for _ in range(input_dim)] for _ in range(output_dim)]
        return TensorT(data)

    @staticmethod
    def xavier_uniform(input_dim, output_dim):
        """
        Xavier (Glorot) uniform initialization.
        """
        limit = math.sqrt(6 / (input_dim + output_dim))
        # return np.random.uniform(-limit, limit, (output_dim, input_dim))
        data = [[random.uniform(-limit, limit) for _ in range(input_dim)] for _ in range(output_dim)]
        return TensorT(data)

    @staticmethod
    def xavier_normal(input_dim, output_dim):
        """
        Xavier (Glorot) normal initialization.
        """
        std = math.sqrt(2 / (input_dim + output_dim))
        # return np.random.normal(0, std, (output_dim, input_dim))
        data = [[random.gauss(0, std) for _ in range(input_dim)] for _ in range(output_dim)]
        return TensorT(data)
    @staticmethod
    def he_uniform(input_dim, output_dim):
        """
        He uniform initialization.
        """
        limit = math.sqrt(6 / input_dim)
        data = [[random.uniform(-limit, limit) for _ in range(input_dim)] for _ in range(output_dim)]
        return TensorT(data)  
    
    @staticmethod
    def he_normal(input_dim, output_dim):
        """
        He normal initialization.
        """
        std = math.sqrt(2 / input_dim)
        # return np.random.normal(0, std, (output_dim, input_dim))
        data = [[random.gauss(0, std) for _ in range(input_dim)] for _ in range(output_dim)]
        return TensorT(data)

    @staticmethod
    def lecun_uniform(input_dim, output_dim):
        """
        LeCun uniform initialization.
        """
        limit = math.sqrt(3 / input_dim)
        # return np.random.uniform(-limit, limit, (output_dim, input_dim))
        data = [[random.uniform(-limit, limit) for _ in range(input_dim)] for _ in range(output_dim)]
        return TensorT(data)

    @staticmethod
    def lecun_normal(input_dim, output_dim):
        """
        LeCun normal initialization.
        """
        std = math.sqrt(1 / input_dim)
        # return np.random.normal(0, std, (output_dim, input_dim))
        data = [[random.gauss(0, std) for _ in range(input_dim)] for _ in range(output_dim)]
        return TensorT(data)

    @staticmethod
    def get(name: str):
        table = {
            'zero': WeightInitializer.zero,
            # 'constant': WeightInitializer.constant,
            'random_normal': WeightInitializer.random_normal,
            'random_uniform': WeightInitializer.random_uniform,
            'xavier_uniform': WeightInitializer.xavier_uniform,
            'xavier_normal': WeightInitializer.xavier_normal,
            'he_uniform': WeightInitializer.he_uniform,
            'he_normal': WeightInitializer.he_normal,
            'lecun_uniform': WeightInitializer.lecun_uniform,
            'lecun_normal': WeightInitializer.lecun_normal,
        }
        return table[name]