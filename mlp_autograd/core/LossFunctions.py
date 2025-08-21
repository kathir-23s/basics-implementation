
from ..tensor.tensor_scratch import TensorT

class LossFunction:

    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        """
        y_true: (C, m) one-hot
        y_pred: (C, m) softmax probs
        """
        y_pred_clipped = y_pred.tclip(1e-15, 1 - 1e-15)
        log_pred = y_pred_clipped.tlog()
        prod = y_true * log_pred
        result = prod.tsum_axis()
        m = y_true.shape[1]
        
        # result.data is already the computed value, just divide by m
        final_loss = -result.tsum() / m  # Get scalar value
        result_data = [[final_loss]]      # Wrap in nested list for TensorT
        
        out = TensorT(result_data, _op='cross_entropy_loss', _parent=(y_true, y_pred))

        def backward_fn(grad_output):
            y_pred_clipped_data = y_pred_clipped.data
            grad_y_true = y_true._apply_elementwise(grad_output[0][0], log_pred.data, lambda g, p: -g * p / m)
            grad_y_pred = y_pred._apply_elementwise(grad_output, y_true.data, y_pred_clipped_data, lambda g, yt, yp: -g * yt / yp / m)
            return (TensorT(grad_y_true), TensorT(grad_y_pred))

        out.backward_fn = backward_fn
        return out

    @staticmethod
    def categorical_cross_entropy_loss(y_true, y_pred):
        y_pred_clipped = y_pred.tclip(1e-15, 1 - 1e-15)
        log_pred = y_pred_clipped.tlog()
        element_wise_product = y_true * log_pred
        sum_result = element_wise_product.tsum()  # This returns scalar
        m = y_true.shape[1]
        
        final_loss = -sum_result / m
        result_data = [[final_loss]]  # Wrap scalar in nested list

        out = TensorT(result_data, _op='categorical_cross_entropy_loss', _parent=(y_true, y_pred))

        def backward_fn(grad_output):
            y_pred_clipped_data = y_pred.tclip(1e-15, 1 - 1e-15).data
            grad_y_true = y_true._apply_elementwise(grad_output[0], log_pred.data, lambda g, log_p: -g * log_p / m)
            grad_y_pred = y_pred._apply_elementwise(grad_output, y_true.data, y_pred_clipped_data, lambda g, yt, yp: -g * yt / yp / m)
            return (TensorT(grad_y_true), TensorT(grad_y_pred))

        out.backward_fn = backward_fn
        return out

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        diff = y_pred - y_true
        squared_diff = diff * diff
        mean_val = squared_diff.tmean()  # Returns scalar
        result_data = [[mean_val]]       # Wrap in nested list

        out = TensorT(result_data, _op='mean_squared_error', _parent=(y_true, y_pred))

        def backward_fn(grad_output):
            total_size = y_true.shape[0] * y_true.shape[1]
            diff_data = (y_pred - y_true).data
            grad_y_true = y_true._apply_elementwise(grad_output, diff_data, lambda g, d: -2 * g * d / total_size)
            grad_y_pred = y_pred._apply_elementwise(grad_output, diff_data, lambda g, d: 2 * g * d / total_size)
            return (TensorT(grad_y_true), TensorT(grad_y_pred))

        out.backward_fn = backward_fn
        return out

    @staticmethod
    def binary_cross_entropy_loss(y_true, y_pred):
        y_pred_clipped = y_pred.tclip(1e-15, 1 - 1e-15)
        ones = TensorT.unit_tensor(1.0, y_true.shape)
        
        log_y_pred = y_pred_clipped.tlog()
        one_minus_y_pred = ones - y_pred_clipped
        log_one_minus_y_pred = one_minus_y_pred.tlog()
        one_minus_y_true = ones - y_true

        term1 = y_true * log_y_pred
        term2 = one_minus_y_true * log_one_minus_y_pred
        sum_terms = term1 + term2
        
        mean_val = -sum_terms.tmean()  # Returns scalar
        result_data = [[mean_val]]     # Wrap in nested list

        out = TensorT(result_data, _op='binary_cross_entropy_loss', _parent=(y_true, y_pred))

        def backward_fn(grad_output):
            y_pred_clipped_data = y_pred.tclip(1e-15, 1 - 1e-15).data
            one_minus_y_pred_data = (ones - y_pred_clipped).data
            total_size = y_true.shape[0] * y_true.shape[1]
            
            grad_y_true = y_true._apply_elementwise(grad_output, y_pred_clipped_data, one_minus_y_pred_data, 
                                                lambda g, yp, omp: -g * (1/yp - 1/omp) / total_size)
            grad_y_pred = y_pred._apply_elementwise(grad_output[0][0], y_true.data, y_pred_clipped_data, one_minus_y_pred_data,
                                                lambda g, yt, yp, omp: -g * (-yt/yp + (1-yt)/omp) / total_size)
            return (TensorT(grad_y_true), TensorT(grad_y_pred))

        out.backward_fn = backward_fn
        return out

    @staticmethod
    def hinge_loss(y_true, y_pred):
        y_true_modified_data = y_true._apply_unary(y_true.data, lambda x: -1 if x == 0 else x)
        y_true_modified = TensorT(y_true_modified_data)

        product = y_true_modified * y_pred
        margin = TensorT.unit_tensor(1.0, product.shape) - product
        hinge_values_data = margin._apply_unary(margin.data, lambda x: max(0, x))
        hinge_values = TensorT(hinge_values_data)
        
        mean_val = hinge_values.tmean()  # Returns scalar
        result_data = [[mean_val]]       # Wrap in nested list

        out = TensorT(result_data, _op='hinge_loss', _parent=(y_true, y_pred))

        def backward_fn(grad_output):
            total_size = y_true.shape[0] * y_true.shape[1]
            product_data = (y_true_modified * y_pred).data
            
            grad_y_true = y_true._apply_elementwise(grad_output, product_data, y_pred.data, 
                                                lambda g, prod, yp: g * yp / total_size if prod < 1 else 0.0)
            grad_y_pred = y_pred._apply_elementwise(grad_output, product_data, y_true_modified_data, 
                                                lambda g, prod, yt: -g * yt / total_size if prod < 1 else 0.0)
            return (TensorT(grad_y_true), TensorT(grad_y_pred))

        out.backward_fn = backward_fn
        return out

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        diff = y_pred - y_true
        abs_diff_data = diff._apply_unary(diff.data, lambda x: abs(x))
        abs_diff = TensorT(abs_diff_data)
        
        mean_val = abs_diff.tmean()  # Returns scalar
        result_data = [[mean_val]]   # Wrap in nested list

        out = TensorT(result_data, _op='mean_absolute_error', _parent=(y_true, y_pred))

        def backward_fn(grad_output):
            total_size = y_true.shape[0] * y_true.shape[1]
            diff_data = (y_pred - y_true).data
            
            grad_y_true = y_true._apply_elementwise(grad_output, diff_data, 
                                                lambda g, d: -g * (1 if d > 0 else -1 if d < 0 else 0) / total_size)
            grad_y_pred = y_pred._apply_elementwise(grad_output, diff_data, 
                                                lambda g, d: g * (1 if d > 0 else -1 if d < 0 else 0) / total_size)
            return (TensorT(grad_y_true), TensorT(grad_y_pred))

        out.backward_fn = backward_fn
        return out



    @staticmethod
    def get(name: str):
        table = {
            'binary_cross_entropy_loss': (LossFunction.binary_cross_entropy_loss, None),
            'categorical_cross_entropy_loss': (LossFunction.categorical_cross_entropy_loss, None),
            'cross_entropy_loss': (LossFunction.cross_entropy_loss, None),
            'mean_squared_error': (LossFunction.mean_squared_error, None),
            'hinge_loss': (LossFunction.hinge_loss, None),
            'mean_absolute_error': (LossFunction.mean_absolute_error, None),
        }
        return table[name]
