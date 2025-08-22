
from tensor.tensor_scratch import TensorT

class LossFunction:

    @staticmethod
    def cross_entropy_loss(y_true, y_pred):

        """
        Multiclass cross-entropy with one-hot labels:
        L = -(1/m) * sum_{i,j} y_true_ij * log(y_pred_ij)
        Expects y_pred to be softmax probabilities (C, m) and y_true one-hot (C, m).
        """
        # ----- forward (unchanged) -----
        eps = 1e-15
        y_pred_clipped = y_pred.tclip(eps, 1 - eps)     # (C, m)
        log_pred = y_pred_clipped.tlog()                # (C, m)
        prod = y_true * log_pred                        # (C, m)
        per_col_sum = prod.tsum_axis()                  # (1, m) or (m,) depending on your impl
        m = y_true.shape[1]
        loss_scalar = -per_col_sum.tsum() / m           # scalar TensorT or python float
        out = TensorT([[loss_scalar]]) if not isinstance(loss_scalar, TensorT) else loss_scalar
        # record parents for backward
        out._op = 'cross_entropy_loss'
        out._parent = (y_true, y_pred)


        def backward_fn(grad_output):
            # treat upstream as scalar 1.0
            go = 1.0
            # grad wrt y_pred (probs): -(Y / clip(S)) / m
            gy_pred = [[0.0 for _ in range(y_pred.shape[1])] for _ in range(y_pred.shape[0])]
            for i in range(y_pred.shape[0]):
                for j in range(y_pred.shape[1]):
                    s = y_pred_clipped.data[i][j]
                    y = y_true.data[i][j]
                    gy_pred[i][j] = -go * (y / s) / m

            # grad wrt y_true (labels): 0 (do not backprop into constants)
            gy_true = [[0.0 for _ in range(y_true.shape[1])] for _ in range(y_true.shape[0])]

            return (TensorT(gy_true), TensorT(gy_pred))

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

        out = TensorT([[final_loss]], _op='categorical_cross_entropy_loss', _parent=(y_true, y_pred))

        def backward_fn(grad_output):
            g = grad_output[0][0]
            yp = y_pred_clipped.data
            
            grad_y_true = y_true._apply_elementwise(g, log_pred.data, lambda gg, logp: -gg * logp / m)
            grad_y_pred = y_pred._apply_elementwise(g, y_true.data , yp, lambda gg, yt, ypi: -gg * yt / ypi / m)

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
        # ----- forward (unchanged math) -----
        y_pred_clipped = y_pred.tclip(1e-15, 1 - 1e-15)
        ones = TensorT.unit_tensor(1.0, y_true.shape)

        log_y_pred = y_pred_clipped.tlog()
        one_minus_y_pred = ones - y_pred_clipped
        log_one_minus_y_pred = one_minus_y_pred.tlog()
        one_minus_y_true = ones - y_true

        term1 = y_true * log_y_pred
        term2 = one_minus_y_true * log_one_minus_y_pred
        sum_terms = term1 + term2

        mean_val = -sum_terms.tmean()           # scalar
        out = TensorT([[mean_val]], _op='binary_cross_entropy_loss', _parent=(y_true, y_pred))

        # ----- backward (cleaned) -----
        def backward_fn(grad_output):
            # scalar upstream for a scalar loss
            go = 1.0

            # reuse forward's numerics
            y_pred_clipped_data = y_pred_clipped.data
            one_minus_y_pred_data = one_minus_y_pred.data

            total_size = y_true.shape[0] * y_true.shape[1]

            # dL/dy_pred  = -( y/yp - (1-y)/(1-yp) ) / N
            grad_y_pred = y_pred._apply_elementwise(
                go, y_true.data, y_pred_clipped_data, one_minus_y_pred_data,
                lambda g, yt, yp, omp: g * (-yt/yp + (1-yt)/omp) / total_size
            )

            # dL/dy_true = 0  (labels are constants)
            gy_true = [[0.0 for _ in range(y_true.shape[1])] for _ in range(y_true.shape[0])]

            return (TensorT(gy_true), TensorT(grad_y_pred))

        out.backward_fn = backward_fn
        return out

    
    @staticmethod
    def hinge_loss(y_true, y_pred):
        """
        Binary hinge loss:
            L = mean( max(0, 1 - y * y_pred) )
        where y in {-1, +1}. If y_true is {0,1}, we internally map y = 2*y_true - 1.
        """
        # ----- forward -----
        # map labels to {-1, +1} WITHOUT creating a gradient path to labels
        # (pure data read)
        y_signed_data = [[2.0 * y_true.data[i][j] - 1.0 for j in range(y_true.shape[1])]
                         for i in range(y_true.shape[0])]
        y_signed = TensorT(y_signed_data)

        # margin = 1 - y * y_pred
        margin = TensorT.unit_tensor(1.0, y_true.shape) - (y_signed * y_pred)
        # loss = mean( relu(margin) )
        zeros = TensorT.unit_tensor(0.0, y_true.shape)
        relu_margin = margin.tmax(zeros)        # elementwise max
        loss_scalar = relu_margin.tmean()       # scalar
        out = TensorT([[loss_scalar]], _op='hinge_loss', _parent=(y_true, y_pred))

        # ----- backward -----
        def backward_fn(grad_output):
            go = 1.0  # scalar upstream for scalar loss
            N = y_true.shape[0] * y_true.shape[1]

            # mask: 1 where margin > 0 else 0  (no grad when correctly classified with margin >= 1)
            mask = [[1.0 if margin.data[i][j] > 0.0 else 0.0
                     for j in range(y_true.shape[1])]
                    for i in range(y_true.shape[0])]

            # dL/dy_pred = (-y) / N  if margin>0  else 0
            grad_y_pred = [[(-y_signed.data[i][j] / N) * mask[i][j]
                            for j in range(y_true.shape[1])]
                           for i in range(y_true.shape[0])]

            # dL/dy_true = 0  (labels are constants; even though math would give -y_pred/N*mask if y were variable)
            grad_y_true = [[0.0 for _ in range(y_true.shape[1])]
                           for _ in range(y_true.shape[0])]

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