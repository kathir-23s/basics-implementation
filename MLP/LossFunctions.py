import numpy as np
class LossFunction:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    @staticmethod
    def mean_squared_error_gradient(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    @staticmethod
    def cross_entropy_gradient(y_true, y_pred):
        return -y_true / (y_pred + 1e-15) / y_true.shape[0]
    
    @staticmethod
    def binary_cross_entropy_loss(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def binary_cross_entropy_gradient(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / y_pred + (1 - y_true) / (1 - y_pred)
    
    @staticmethod
    def hinge_loss(y_true, y_pred):
        y_true = np.where(y_true == 0, -1, y_true)
        return np.mean(np.maximum(0, 1 - y_true * y_pred))
    
    @staticmethod
    def hinge_loss_gradient(y_true, y_pred):   
        y_true = np.where(y_true == 0, -1, y_true)
        grad = np.where(y_pred * y_true < 1, -y_true, 0)
        return grad / y_true.size
    
    @staticmethod
    def categorical_cross_entropy_loss(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    @staticmethod
    def categorical_cross_entropy_gradient(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / (y_pred + 1e-15) / y_true.shape[0]
    
    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def mean_absolute_error_gradient(y_true, y_pred):
        return np.sign(y_pred - y_true) / y_true.size