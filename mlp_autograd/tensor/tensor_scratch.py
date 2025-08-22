# Custom tensor class in python without any dependency (not even numpy)
import random
import math

class TensorT:

    def __init__(self, data, _op=None, _parent=()): #, req_grad=False):
        if isinstance(data, list) and data and not isinstance(data[0], list):
            data = [data]
        self._check_rectangular(data)
        self.data = data
        self.shape = self._get_shape(data)
        if len(self.shape) != 2:
            raise ValueError("Supporting upto 2D Tensors (Matrices) for now")

        self.grad = None
        self._op = _op
        self._parent = _parent
        self.backward_fn = None


    def _check_rectangular(self, data):
        """Recursively ensure all sublists have the same length."""
        if all(isinstance(x, list) for x in data):
            first_len = len(data[0])
            for sub in data:
                if len(sub) != first_len:
                    raise ValueError("Ragged tensor: inconsistent sublist lengths")
                self._check_rectangular(sub)

    def _get_shape(self, data):
        if isinstance(data, list):
            if len(data) == 0:
                return (0,)
            return (len(data), ) + self._get_shape(data[0])
        else:
            return ()
         
    def __repr__(self):
        if len(self.shape) == 2:
            rows = ",\n ".join(str(row) for row in self.data)
            return f"tensor:\n[{rows}], shape: {self.shape}"
        else:
            # Fallback for non-2D tensors
            return f"tensor: {self.data}, shape: {self.shape}"
    
# OPERATIONS
    def _elementwise_op(self, other, op):
        other_data = other.data if isinstance(other, TensorT) else other
        result_shape = self._broadcast_shape(self.shape, other.shape if isinstance(other, TensorT) else ())
        self_broadcasted = self._broadcast_to(self.data, self.shape, result_shape)
        other_broadcasted = other_data if not isinstance(other, TensorT) else self._broadcast_to(other_data, other.shape, result_shape)
        result = self._apply_elementwise(self_broadcasted, other_broadcasted, op)
        
        return result

    def _apply_elementwise(self, *args):
        """
        Apply a function elementwise across multiple tensors/scalars.
        The last argument must be the function.
        Supports broadcasting like the old 2-arg version.
        """
        *arrays, op = args  # separate operands and function

        # Convert TensorT → raw data
        arrays = [arr.data if isinstance(arr, TensorT) else arr for arr in arrays]

        def recurse(*vals):
            # Base case: all scalars
            if all(not isinstance(v, list) for v in vals):
                return op(*vals)

            # Handle broadcasting: if some vals are scalars, expand them
            max_len = max(len(v) if isinstance(v, list) else 1 for v in vals)

            expanded = []
            for v in vals:
                if not isinstance(v, list):  # scalar → repeat
                    expanded.append([v] * max_len)
                elif len(v) == 1 and max_len > 1:  # length-1 list → broadcast
                    expanded.append(v * max_len)
                else:
                    expanded.append(v)

            # Recurse elementwise
            return [recurse(*items) for items in zip(*expanded)]

        return recurse(*arrays)

    
    def _broadcast_shape(self, shape1, shape2):
        '''
        Broadcasting when elementwise operations are performed 
        between two tensors of different sizes
        '''
        result = []
        for i in range(max(len(shape1), len(shape2))):
            dim1 = shape1[-1 - i] if i < len(shape1) else 1
            dim2 = shape2[-1 - i] if i < len(shape2) else 1
            
            if dim1 == dim2 or dim1 == 1 or dim2 == 1:
                result.append(max(dim1, dim2))
            else:
                raise ValueError(f"Shapes {shape1} and {shape2} not broadcastable")
        return tuple(reversed(result))

    def _broadcast_to(self, data, from_shape, to_shape):
        '''
        Broadcasting when elementwise operations are performed 
        between two tensors of different sizes
        '''
        # Recursively replicate data to match to_shape
        if len(to_shape) == 0:
            return data  # scalar
        if len(from_shape) < len(to_shape):
            from_shape = (1,) * (len(to_shape) - len(from_shape)) + from_shape
        if from_shape[0] == to_shape[0]:
            # Broadcast each sublist
            return [self._broadcast_to(d, from_shape[1:], to_shape[1:]) for d in data]
        elif from_shape[0] == 1:
            # Repeat the same sublist to match size
            return [self._broadcast_to(data[0], from_shape[1:], to_shape[1:]) for _ in range(to_shape[0])]
        else:
            # Should not reach here if shapes check succeeded
            raise ValueError("Incompatible shapes during broadcasting")
    
    def _apply_unary(self, a, op):
        if not isinstance(a, list):
            return op(a)
        return [self._apply_unary(x, op) for x in a]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __add__(self, other):
        result = self._elementwise_op(other, lambda x,y: x+y)
        out = TensorT(result, _op='add', _parent=(self, other))

        def backward_fn(grad_op):
            grad_self = grad_op
            grad_other = grad_op

            return grad_self, grad_other
        
        out.backward_fn = backward_fn        
        return out

    
    def __mul__(self, other):
        result = self._elementwise_op(other, lambda x,y: x*y)
        out = TensorT(result, _op='mul', _parent=(self, other))

        def backward_fn(grad_op):
            
            grad_self = self._apply_elementwise(grad_op,
            other.data if isinstance(other, TensorT) else other,
            lambda x, y: x * y)
            grad_other = other._apply_elementwise(grad_op,
            self.data if isinstance(self, TensorT) else self,
            lambda x, y: x * y)

            return grad_self, grad_other

        out.backward_fn = backward_fn
        return out

    def __sub__(self, other):
        result =  self._elementwise_op(other, lambda x,y: x-y)
        out = TensorT(result, _op='sub', _parent=(self, other))

        def backward_fn(grad_op):

            grad_self = grad_op
            grad_other = other._apply_unary(grad_op, lambda x: -x)
            return grad_self, grad_other

        out.backward_fn = backward_fn
        return out
    
    def __truediv__(self, other):
    # Compute elementwise division
        result = self._elementwise_op(other, lambda x, y: x / y)
        out = TensorT(result, _op='div', _parent=(self, other))

        def backward_fn(grad_output):
            # grad w.r.t self: grad_output / other
            grad_self = self._apply_elementwise(
                grad_output,
                other.data if isinstance(other, TensorT) else other,
                lambda x, y: x / y
            )
            # grad w.r.t other: -grad_output * self / (other^2)
            grad_other = other._apply_elementwise(
                grad_output,
                self.data,
                lambda go, s: -go * s
            )
            # Multiply grad_other by 1/(other^2)
            grad_other = other._apply_elementwise(
                grad_other,
                self._apply_elementwise(
                    other.data if isinstance(other, TensorT) else other,
                    other.data if isinstance(other, TensorT) else other,
                    lambda x, y: x * y
                ),
                lambda x, y: x / y
            )
            return grad_self, grad_other

        out.backward_fn = backward_fn
        return out
    
    def __neg__(self):
        return TensorT(self._apply_unary(self.data, lambda x: -x))
    
    def __pow__(self, other):
        return TensorT(self._apply_unary(self.data, lambda x : x**other))
    
    def tlog(self):
        """Element-wise natural logarithm."""
        return TensorT(self._apply_unary(self.data, math.log))

    def texp(self):
        """Element-wise exponential."""
        return TensorT(self._apply_unary(self.data, math.exp))
    
    def tsum(self):
        """Return the sum of all elements."""
        return sum(item for row in self.data for item in row)

    def tmean(self):
        """Return the mean (average) of all elements."""
        total_elements = self.shape[0] * self.shape[1]
        return self.tsum() / total_elements if total_elements > 0 else float('nan')

    
# DEFINING RANDOM TENSORS
    @classmethod
    def unit_tensor(cls, unit: float, shape):
        """Create a tensor filled with ones or zeros."""
        if unit not in (0, 1):
            raise ValueError("unit must be 0 or 1")
        unit = float(unit)  # ensure float type
        def build(s):
            if len(s) == 1:
                return [unit] * s[0]
            return [build(s[1:]) for _ in range(s[0])]
        return cls(build(shape))
    
    @classmethod
    def const_tensor(cls, unit: float, shape):
        """Create a tensor filled with ones or zeros."""
        unit = float(unit)  # ensure float type
        def build(s):
            if len(s) == 1:
                return [unit] * s[0]
            return [build(s[1:]) for _ in range(s[0])]
        return cls(build(shape))
    
    @classmethod
    def random_tensor(cls, shape):
        '''Creating a tensor with random values'''
        m, n = shape
        flat = [random.random() for _ in range(m * n)]
        # Reshape into m rows
        data = [flat[i * n : (i + 1) * n] for i in range(m)]
        return cls(data)
        

# MATRIX OPERATIONS
    def tmatmul(self, other):
        
        assert isinstance(self, TensorT) and isinstance(other, TensorT), "Not a tensor"
        assert len(self.shape) == 2 and len(other.shape) == 2, "Not a matrix" # For UPTO 2d tensors
        if self.shape[1] != other.shape[0]:
            raise ValueError("Cannot multiply, order not compatible")
        else:    
            result = [
            [sum(self.data[i][k] * other.data[k][j] for k in range(self.shape[1]))
             for j in range(other.shape[1])]
            for i in range(self.shape[0])
        ]
        
        out = TensorT(result, _op='matmul', _parent=(self, other))

        def backward_fn(grad_op):
            # grad w.r.t self: grad_op * other^T
            grad_self = TensorT(grad_op).tmatmul(other.ttranspose())
            # grad w.r.t other: self^T * grad_op
            grad_other = self.ttranspose().tmatmul(TensorT(grad_op))
            return grad_self.data, grad_other.data

        out.backward_fn = backward_fn
        return out

    def block_matmul(self, other, block_size):
        """
        Blocked matrix multiply supporting rectangular shapes.

        A: self.data, shape (m, k)
        B: other.data, shape (k, n)
        C: result,     shape (m, n)
        """
        assert isinstance(self, TensorT) and isinstance(other, TensorT), "Both must be TensorT"
        assert len(self.shape) == 2 and len(other.shape) == 2, "Both tensors must be 2D matrices"
        m, k = self.shape
        k2, n = other.shape
        if k != k2:
            raise ValueError(f"Incompatible shapes: {self.shape} @ {other.shape}")

        # Output
        C = [[0.0 for _ in range(n)] for _ in range(m)]

        # Blocked triple-loop: ii over rows of A/C, jj over cols of B/C, kk over shared dim
        for ii in range(0, m, block_size):
            iimax = min(ii + block_size, m)
            for jj in range(0, n, block_size):
                jjmax = min(jj + block_size, n)
                for kk in range(0, k, block_size):
                    kkmax = min(kk + block_size, k)
                    # Compute C[i,j] partial sums for this block
                    for i in range(ii, iimax):
                        Ai = self.data[i]
                        Ci = C[i]
                        for kk2 in range(kk, kkmax):
                            aik = Ai[kk2]
                            Bk = other.data[kk2]
                            # accumulate aik * Bk[j] for j block
                            for j in range(jj, jjmax):
                                Ci[j] += aik * Bk[j]

        out = TensorT(C, _op='block_matmul', _parent=(self, other))

        def backward_fn(grad_out):
            # grad_out has shape (m, n)
            # dL/dA = grad_out @ B^T  → shape (m, k)
            # dL/dB = A^T @ grad_out  → shape (k, n)
            grad_self = TensorT(grad_out).block_matmul(other.ttranspose(), block_size)
            grad_other = self.ttranspose().block_matmul(TensorT(grad_out), block_size)
            return grad_self.data, grad_other.data

        out.backward_fn = backward_fn
        return out

    def tmatmul_fast(self, other):
        """
        Pure-Python cache-friendlier matmul using your own methods.
        (m x k) @ (k x n) -> (m x n)
        Strategy: use other.ttranspose() once; then inner loop is a dot over two flat rows.
        """
        assert isinstance(self, TensorT) and isinstance(other, TensorT), "otheroth must othere TensorT"
        assert len(self.shape) == 2 and len(other.shape) == 2, "otheroth tensors must othere 2D"
        m, k = self.shape
        k2, n = other.shape
        if k != k2:
            raise ValueError(f"Incompatiotherle shapes: {self.shape} @ {other.shape}")

        other_t = other.ttranspose()             # (n x k) — your method
        C = [[0.0 for _ in range(n)] for _ in range(m)]

        for i in range(m):
            Ai = self.data[i]               # length k
            Ci = C[i]
            for j in range(n):
                otherj = other_t.data[j]         # row j of other^T -> col j of other, length k
                s = 0.0
                for kk in range(k):
                    s += Ai[kk] * otherj[kk]
                Ci[j] = s

        out = TensorT(C, _op='matmul_fast_pure', _parent=(self, other))

        def backward_fn(grad_op):
            # dA = G @ other^T ; dother = A^T @ G   (use same fast kernel + your transposes)
            dA = TensorT.matmul_fast(TensorT(grad_op), other.ttranspose()).data
            dB = TensorT.matmul_fast(self.ttranspose(), TensorT(grad_op)).data
            return dA, dB

        out.backward_fn = backward_fn
        return out



    def ttranspose(self):
        '''Creating Transpose of the tensor
        
        input: TensorT of dimension 2 (shape: row, column)
        output: TensorT of dimension 2 (shape: column, row)

        Workings:
        The i loop -> will populate the new tensor's inner list with len(row)
        The j loop -> will iterate to num of columns

        [[a, b, c], [d, e, f]] --> transpose --> [[a, d], [b, e], [c, f]]
        i will populate inner list with m times
        j will initiate creating n inner lists
        '''

        row, col = self.shape
        tranposed_tensor = [
            [self.data[i][j] for i in range(row)]  
            for j in range(col)]
        
        return TensorT(tranposed_tensor)

    def tflatten(self):
        '''
        This will return a flat list of all the elements in the tensor
        
        Input: TensorT of shape(mxn)
        Output: List of size (m*n) --> vector of size 1x(m*n)'''
        # m,n = self.shape
        flat_tensor = [item for row in self.data for item in row]
        return flat_tensor
  
    def treshape(self, new_shape: tuple):
        '''
        This will reshape the tensor to another tensor with compatible shape

        Input:
        TensorT of shape (m x n)
        new shape: Tuple (a x b)

        Condition: m*n == a*b (number of element must be equal)

        Output:
        TensorT: shape (a x b)
        '''
        m, n = self.shape
        new_m, new_n = new_shape

        if m*n != new_m*new_n:
            raise ValueError(
            f"Incompatible Size for reshape. "
            f"New size {new_m, new_n} should have {m * n} elements"
        )
        flat = self.tflatten()

        reshaped_tensor = [flat[i* new_n:(i+1) * new_n]
                           for i in range(new_m)]
            
        return TensorT(reshaped_tensor)

    def tsum_axis(self, axis=1, keepdims=True):
        """
        Sum along specified axis
        axis=0: sum along rows (column-wise sum) 
        axis=1: sum along columns (row-wise sum)
        """
        if axis == 0:
            # Sum along rows: (2,3) → (1,3) if keepdims else (3,)
            result = [sum(self.data[i][j] for i in range(self.shape[0])) 
                    for j in range(self.shape[1])]
            return TensorT([result]) if keepdims else TensorT([result])
        
        elif axis == 1:
            # Sum along columns: (2,3) → (2,1) if keepdims else (2,)  
            if keepdims:
                result = [[sum(row)] for row in self.data]
            else:
                result = [sum(row) for row in self.data]
            return TensorT(result)

    def tmaximum(self, scalar):
        """Element-wise maximum with scalar (for ReLU)"""
        result = self._apply_unary(self.data, lambda x: max(x, scalar))
        return TensorT(result)
    
    def tclip(self, min_val, max_val):
        """Clip values between min_val and max_val (for numerical stability)"""
        result = self._apply_unary(self.data, lambda x: max(min_val, min(max_val, x)))
        return TensorT(result)
    
    @classmethod
    def from_numpy(cls, np_array):
        """Convert numpy array to TensorT"""
        if np_array.ndim == 1:
            return cls([np_array.tolist()])
        elif np_array.ndim == 2:
            return cls(np_array.tolist())
        else:
            raise ValueError("Only 1D and 2D numpy arrays supported")

    def to_numpy(self):
        """Convert TensorT to numpy array"""
        import numpy as np
        return np.array(self.data)


# MLP METHODS - BACKWARD PASS
    def zero_grad(self):
        """Zero out gradients for this tensor and all tensors in the computational graph"""
        visited = set()
        
        def _zero(tensor):
            if id(tensor) in visited:
                return
            visited.add(id(tensor))
            tensor.grad = None
            
            # Traverse to parent tensors
            for parent in tensor._parent:
                if isinstance(parent, TensorT):
                    _zero(parent)
        
        _zero(self)

    def backward(self, grad=None):
        if grad is None:
            if self.shape == ():
                grad = 1.0
            else:
                grad = TensorT.unit_tensor(1.0, self.shape).data

        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self._apply_elementwise(self.grad, grad, lambda x, y: x + y)

        if not self._parent or self.backward_fn is None:
            return

        parent_grads = self.backward_fn(grad)
        for parent, parent_grad in zip(self._parent, parent_grads):
            if isinstance(parent, TensorT):
                # if parent.grad is None:
                #     parent.grad = parent_grad
                # else:
                #     parent.grad = parent._apply_elementwise(parent.grad, parent_grad, lambda x, y: x + y)
                parent.backward(grad=parent_grad)
            else:
                raise ValueError("Parent must be a TensorT instance")