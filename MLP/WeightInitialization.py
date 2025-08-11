import numpy as np
class WeightInitializer:

    @staticmethod
    def zero(input_dim, output_dim):
        return np.zeros((output_dim, input_dim))

    @staticmethod
    def constant(input_dim, output_dim, value=0.01):
        return np.full((output_dim, input_dim), value)
    
    # @staticmethod
    # def random(input_dim, output_dim):    
    #     return np.random.rand(output_dim, input_dim) * 2 - 1
    
    @staticmethod
    def random_uniform(input_dim, output_dim, low=-0.05, high=0.05):
        return np.random.uniform(low, high, (output_dim, input_dim))

    @staticmethod
    def random_normal(input_dim, output_dim, mean=0.0, std=0.05):
        return np.random.normal(mean, std, (output_dim, input_dim))

    @staticmethod
    def xavier_uniform(input_dim, output_dim):
        limit = np.sqrt(6 / (input_dim + output_dim))
        return np.random.uniform(-limit, limit, (output_dim, input_dim))

    @staticmethod
    def xavier_normal(input_dim, output_dim):
        std = np.sqrt(2 / (input_dim + output_dim))
        return np.random.normal(0, std, (output_dim, input_dim))

    @staticmethod
    def he_uniform(input_dim, output_dim):
        limit = np.sqrt(6 / input_dim)
        return np.random.uniform(-limit, limit, (output_dim, input_dim))

    @staticmethod
    def he_normal(input_dim, output_dim):
        std = np.sqrt(2 / input_dim)
        return np.random.normal(0, std, (output_dim, input_dim))

    @staticmethod
    def lecun_uniform(input_dim, output_dim):
        limit = np.sqrt(3 / input_dim)
        return np.random.uniform(-limit, limit, (output_dim, input_dim))

    @staticmethod
    def lecun_normal(input_dim, output_dim):
        std = np.sqrt(1 / input_dim)
        return np.random.normal(0, std, (output_dim, input_dim))
    
    