
import math
from ..tensor.tensor_scratch import TensorT

class ActivationFunction:
    
    @staticmethod
    def sigmoid(z):
        """Computes the sigmoid of z."""
        z_clip = z.tclip(-500, 500)
        exp_neg_z = (-z_clip).texp()
        ones = TensorT.unit_tensor(1.0, exp_neg_z.shape)
        result_data = (ones / (ones + exp_neg_z)).data

        out = TensorT(result_data, _op='sigmoid', _parent=(z,))

        def backward_fn(grad_op):
            s_data = result_data
            grad_self = z._apply_elementwise(grad_op, s_data, lambda x, y: x * y * (1 - y))
            return (grad_self,)  # ✅ Return tuple

        out.backward_fn = backward_fn
        return out

    @staticmethod
    def relu(z):
        """Computes the Rectified Linear Unit of z."""
        result_data = z._apply_unary(z.data, lambda x: max(0, x))
        out = TensorT(result_data, _op='relu', _parent=(z,))

        def backward_fn(grad_op):
            grad_self = z._apply_elementwise(grad_op, z.data, lambda g, x: g if x > 0 else 0)
            return (grad_self,)  # ✅ Return tuple

        out.backward_fn = backward_fn
        return out


    @staticmethod
    def tanh(z):
        """Computes the hyperbolic tangent of z."""
        two_x = z * 2
        exp_2x = two_x.texp()
        ones = TensorT.unit_tensor(1.0, exp_2x.shape)
        tanh_data = ((exp_2x - ones) / (exp_2x + ones)).data

        out = TensorT(tanh_data, _op='tanh', _parent=(z,))

        def backward_fn(grad_op):
            grad_self = z._apply_elementwise(grad_op, tanh_data, lambda g, y: g * (1 - y ** 2))
            return (grad_self,)  # ✅ Return tuple

        out.backward_fn = backward_fn
        return out


    @staticmethod
    def softmax(z):
        """Computes the softmax of z."""
        # Find max for numerical stability (column-wise for each sample)
        z_max_per_col = [max(col) for col in zip(*z.data)]
        
        # Subtract max for stability - manually implement broadcasting
        z_stable_data = []
        for i, row in enumerate(z.data):
            stable_row = [x - z_max_per_col[j] for j, x in enumerate(row)]
            z_stable_data.append(stable_row)
        
        # Compute exponentials
        exp_data = z._apply_unary(z_stable_data, math.exp)
        
        # Compute column sums for normalization
        sum_exp_per_col = [sum(col) for col in zip(*exp_data)]
        
        # Normalize - divide each element by its column sum
        result_data = []
        for i, row in enumerate(exp_data):
            normalized_row = [x / sum_exp_per_col[j] for j, x in enumerate(row)]
            result_data.append(normalized_row)

        out = TensorT(result_data, _op='softmax', _parent=(z,))

        def backward_fn(grad_op):
            # Softmax derivative: s * (δ - s) where δ is kronecker delta
            # For simplicity, using element-wise: s * (1 - s) * grad
            s_data = result_data
            grad_self = z._apply_elementwise(grad_op, s_data, lambda g, s_val: g * s_val * (1 - s_val))
            return (grad_self,)  # ✅ Return tuple

        out.backward_fn = backward_fn
        return out


    @staticmethod
    def leaky_relu(z, alpha=0.01):
        """Computes the Leaky ReLU of z as single semantic operation"""
        result_data = z._apply_unary(z.data, lambda x: x if x > 0 else alpha * x)
        out = TensorT(result_data, _op='leaky_relu', _parent=(z,))

        def backward_fn(grad_op):
            grad_self = z._apply_elementwise(grad_op, z.data, lambda g, x: g * (1.0 if x > 0 else alpha))
            return (grad_self,)  # ✅ Return tuple

        out.backward_fn = backward_fn
        return out


    @staticmethod
    def elu(z, alpha=1.0):
        """Computes the Exponential Linear Unit of z as single semantic operation"""
        result_data = z._apply_unary(z.data, lambda x: x if x >= 0 else alpha * (math.exp(x) - 1))
        out = TensorT(result_data, _op='elu', _parent=(z,))

        def backward_fn(grad_op):
            grad_self = z._apply_elementwise(grad_op, z.data, lambda g, x: g * (1.0 if x >= 0 else alpha * math.exp(x)))
            return (grad_self,)  # ✅ Return tuple

        out.backward_fn = backward_fn
        return out


    @staticmethod
    def swish(z, beta=1.0):
        """Computes the Swish activation function."""
        beta_z = z * beta
        sig_result = ActivationFunction.sigmoid(beta_z)
        result_data = z._apply_elementwise(z.data, sig_result.data, lambda x, y: x * y)

        out = TensorT(result_data, _op='swish', _parent=(z,))

        def backward_fn(grad_op):
            sig_data = sig_result.data
            grad_self = z._apply_elementwise(grad_op, z.data, sig_data, lambda g, x, s: g * (s + beta * x * s * (1 - s)))
            return (grad_self,)  # ✅ Return tuple

        out.backward_fn = backward_fn
        return out


    @staticmethod
    def softplus(z):
        """Computes the Softplus activation function."""
        result_data = z._apply_unary(z.data, lambda x: math.log(1 + math.exp(-abs(x))) + max(x, 0))
        out = TensorT(result_data, _op='softplus', _parent=(z,))
    
        def backward_fn(grad_op):
            grad_self = z._apply_elementwise(grad_op, z.data, lambda g, x: g * (1 / (1 + math.exp(-x))))
            return (grad_self,)  # ✅ Return tuple
    
        out.backward_fn = backward_fn
        return out
    
    @staticmethod
    def identity(z):
        """Identity activation function."""
        result_data = z.data
        out = TensorT(result_data, _op='identity', _parent=(z,))

        def backward_fn(grad_op):
            grad_self = grad_op  # Identity derivative is 1
            return (grad_self,)  # ✅ Return tuple

        out.backward_fn = backward_fn
        return out


    
    @staticmethod
    def get(name: str):
        table = {
            'relu': (ActivationFunction.relu, None),
            'sigmoid': (ActivationFunction.sigmoid, None),
            'tanh': (ActivationFunction.tanh, None),
            'leaky_relu': (ActivationFunction.leaky_relu, None),
            'softmax': (ActivationFunction.softmax, None),
            'elu': (ActivationFunction.elu, None),
            'swish': (ActivationFunction.swish, None),
            'softplus': (ActivationFunction.softplus, None),
            'identity': (ActivationFunction.identity, None),
        }
        return table[name]

    @staticmethod
    def output_for_loss(loss_name: str): 
        mapping = {
            'binary_cross_entropy_loss': 'sigmoid',
            'categorical_cross_entropy_loss': 'softmax',
            'cross_entropy_loss': 'softmax',
            'mean_squared_error': 'identity',
            'mean_absolute_error': 'identity',
            'hinge_loss': 'identity',
        }
        return ActivationFunction.get(mapping[loss_name])