# Custom tensor class in python without any dependency (not even numpy)
import random
import math

class TensorT:

    def __init__(self, data): #, req_grad=False):
        if isinstance(data, list) and data and not isinstance(data[0], list):
            data = [data]
        self._check_rectangular(data)
        self.data = data
        self.shape = self._get_shape(data)
        if len(self.shape) != 2:
            raise ValueError("Supporting upto 2D Tensors (Matrices) for now")

    def _check_rectangular(self, data):
        """Recursively ensure all sublists have the same length."""
        # if isinstance(data, list):
        #     # All elements must be same type (either list or scalar)
        #     if not all(isinstance(x, (int, float)) or isinstance(x, list) for x in data):
        #         raise ValueError("Data mismatch in the inner list")
            
            # All sublists must have same length
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
        
        return TensorT(result)

    
    def _apply_elementwise(self, a, b, op):
        if not isinstance(a, list) and not isinstance(b, list):
            return op(a,b)
        elif not isinstance(a, list):  # broadcast scalar a
            return [self._apply_elementwise(a, y, op) for y in b]
        elif not isinstance(b, list):  # broadcast scalar b
            return [self._apply_elementwise(x, b, op) for x in a]
        
        return [self._apply_elementwise(x,y,op) for x,y in zip(a,b)]
    
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
        return self._elementwise_op(other, lambda x,y: x+y)
    
    def __mul__(self, other):
        return self._elementwise_op(other, lambda x,y: x*y)
    
    def __sub__(self, other):
        return self._elementwise_op(other, lambda x,y: x-y)
    
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
    def random_tensor(cls, shape):
        '''Creating a tensor with random values'''
        m, n = shape
        flat = [random.random() for _ in range(m * n)]
        # Reshape into m rows
        data = [flat[i * n : (i + 1) * n] for i in range(m)]
        return cls(data)
        

# MATRIX OPERATIONS
    def tmatmul(self, other):
        '''
        This function will return a multiplied matrix of dimension 2

        Input: Tensor1 - 2d - MxN; Tensor1 - 2d - NxK 
        Output: TensorT - 2d - MxK

        Conditions:
        Number of columns in matrix A should be equal to number of rows in matrix B
        '''

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
        
        return TensorT(result)

    def block_matmul(self, other, block_size):
        '''
        Blocked matrix multiplication (cache-friendly optimization).

        Input:
            self  -> TensorT of shape (N, N)
            other -> TensorT of shape (N, N)
            block_size -> integer block dimension (e.g., 32, 64)
        Output:
            TensorT of shape (N, N)

        Conditions:
            - Both operands must be 2D square matrices of the same shape
            - block_size must divide the dimension or be smaller than N
        '''
        # Type checks
        assert isinstance(self, TensorT) and isinstance(other, TensorT), "Both must be TensorT"
        # Shape checks
        assert len(self.shape) == 2 and len(other.shape) == 2, "Both tensors must be 2D matrices"
        # Square matrix check (for simplicity in blocked version)
        assert self.shape[0] == self.shape[1] == other.shape[0] == other.shape[1], \
            "Block multiplication currently supports only square matrices of same dimension"

        n = self.shape[0]  # matrix size
        C = [[0.0 for _ in range(n)] for _ in range(n)]

        # Perform multiplication in blocks
        for ii in range(0, n, block_size):
            for jj in range(0, n, block_size):
                for kk in range(0, n, block_size):
                    for i in range(ii, min(ii + block_size, n)):
                        for j in range(jj, min(jj + block_size, n)):
                            temp_sum = C[i][j]
                            for k in range(kk, min(kk + block_size, n)):
                                temp_sum += self.data[i][k] * other.data[k][j]
                            C[i][j] = temp_sum

        return TensorT(C)

    
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
        flat = self.flatten()

        reshaped_tensor = [flat[i* new_n:(i+1) * new_n]
                           for i in range(new_m)]
            
        return TensorT(reshaped_tensor)
