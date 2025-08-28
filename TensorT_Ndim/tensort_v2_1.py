# Custom tensor class in python without any dependency (not even numpy)
import random
import math

class TensorT:

    def __init__(self, data, _op=None, _parent=()): #, req_grad=False):

        # --- Type validation --- #
        def _is_seq(x):
            return isinstance(x, (list, tuple))
        
        def _validate_numeric(x):
            if _is_seq(x):
                for y in x:
                    _validate_numeric(y)
            else:
                if not isinstance(x, (int, float, bool)):
                    raise TypeError(f"Unsupported dtype {type(x).__name__}; only int/float allowed")

        _validate_numeric(data)

        # --- Shape validation --- #
        shape = self._get_shape(data)

        # --- Rectangularity check ---#
        if len(shape) >= 2:
            self._check_rectangular(data)

        # ---- dtype decision & casting ----
        def _cast_to_float32(x):
            if isinstance(x, (list, tuple)):
                return [_cast_to_float32(y) for y in x]
            return float(x)

        casted = _cast_to_float32(data)
        dtype = "float32"


        # ---- store core fields ----
        self.data = casted
        self.shape = shape
        self.dtype = dtype

        self.grad = None
        self._op = _op
        self._parent = _parent
        self.backward_fn = None

    def _check_rectangular(self, data):
        """Recursively ensure all sublists have the same length."""
        if not isinstance(data, (list, tuple)):
            return
        
        if len(data) == 0:
            return
        
        are_sq = [isinstance(X, (list, tuple)) for X in data]
        if any(are_sq) and not all(are_sq):
            raise ValueError("Inconsistent nesting: mixed list/element at same level")
        
        if all(not a for a in are_sq):
            return
        
        first_len = len(data[0])
        for sub in data:
            if not isinstance(sub, (list, tuple)):
                raise ValueError("Inconsistent nesting: mixed list/element at same level")
            if len(sub) != first_len:
                raise ValueError("Inconsistent sublist lengths")
            self._check_rectangular(sub)

    def _get_shape(self, data):
        if isinstance(data, (list, tuple)):
            return (len(data),) + self._get_shape(data[0]) if len(data) > 0 else (0,)
        else:
            return ()

    def __repr__(self):
        """Pretty-print tensor in NumPy/PyTorch style with truncation and metadata."""

        threshold = 6     # max items per dimension before truncating
        edge_items = 3    # how many items to show on each side if truncating

        def format_scalar(x):
            # Always print as float32 (since all tensors are cast in __init__)
            return f"{float(x):.2f}"

        def format_array(arr, depth=0):
            if not isinstance(arr, (list, tuple)):
                return format_scalar(arr)

            n = len(arr)
            if n == 0:
                return "[]"

            # Truncate if too long
            if n > threshold:
                shown = arr[:edge_items] + ["..."] + arr[-edge_items:]
            else:
                shown = arr

            formatted = []
            for elem in shown:
                if elem == "...":
                    formatted.append("...")
                else:
                    formatted.append(format_array(elem, depth + 1))

            if isinstance(arr[0], (list, tuple)):
                # Multi-dim: align rows/blocks on new lines
                indent = " " * (depth * 4)
                inner = (",\n" + indent + " ").join(formatted)
                return "[" + inner + "]"
            else:
                # 1D: keep inline
                return "[" + ", ".join(formatted) + "]"

        # Scalar case
        if len(self.shape) == 0:
            content = format_scalar(self.data)
            return f"tensor({content}, shape={self.shape}, dtype=float32)"

        # Vector, matrix, higher-rank
        content = format_array(self.data, depth=1)
        return f"tensor({content}, shape={self.shape}, dtype=float32)"



# OPERATIONS
    def _elementwise_op(self, other, op):
        """
        Forward-only elementwise op between self and other (TensorT or scalar).
        Applies NumPy-style broadcasting and returns raw nested-list data.
        """
        # Normalize other → (data, shape)
        if isinstance(other, TensorT):
            other_data, other_shape = other.data, other.shape
        else:
            other_data, other_shape = other, ()

        # Compute common shape and broadcast both operands
        out_shape = self._broadcast_shape(self.shape, other_shape)
        a_b = self._broadcast_to(self.data,  self.shape,  out_shape)
        b_b = self._broadcast_to(other_data, other_shape, out_shape)

        # Apply elementwise op and return raw data
        return self._apply_elementwise(a_b, b_b, op)

    def _unbroadcast(self, grad, from_shape, target_shape):
        """
        Reduce `grad` (shape = from_shape) down to `target_shape` by summing over
        axes that were broadcast in forward.
        """
        if from_shape == target_shape:
            return grad

        orig_target_shape = target_shape

        # Pad target with leading 1s to align ranks
        if len(target_shape) < len(from_shape):
            target_shape = (1,) * (len(from_shape) - len(target_shape)) + target_shape

        def sum_axis(data, axis):
            if axis == 0:
                if not isinstance(data, list):
                    return data
                if len(data) == 0:
                    return []
                if not isinstance(data[0], list):
                    return sum(data)
                transposed = list(zip(*data))
                return [sum_axis(list(group), 0) for group in transposed]
            else:
                if not isinstance(data, list):
                    return data
                return [sum_axis(sub, axis - 1) for sub in data]

        # Reduce axes where target dim is 1 but grad dim > 1
        current, cur_shape = grad, from_shape
        axis = 0
        while axis < len(cur_shape):
            if target_shape[axis] == 1 and cur_shape[axis] > 1:
                current = sum_axis(current, axis)
                cur_shape = cur_shape[:axis] + (1,) + cur_shape[axis + 1:]
            else:
                axis += 1

        # Squeeze leading size-1 axes added by padding to match original target rank
        def squeeze_leading_axis0(data):
            return data[0] if isinstance(data, list) and len(data) == 1 else data

        while len(cur_shape) > len(orig_target_shape):
            # leading padded axes are guaranteed to be 1 here
            current = squeeze_leading_axis0(current)
            cur_shape = cur_shape[1:]

        return current


    def _apply_elementwise(self, *args):
        """
        Apply a function elementwise across multiple tensors/scalars.
        The last argument must be the function.
        Supports NumPy-style broadcasting per axis across all operands.
        Always returns nested Python lists.
        """
        *arrays, op = args  # separate operands and function

        # Convert TensorT → raw data
        arrays = [arr.data if isinstance(arr, TensorT) else arr for arr in arrays]

        def is_seq(x):
            return isinstance(x, (list, tuple))

        def recurse(*vals):
            # Base case: all scalars (non-sequences)
            if all(not is_seq(v) for v in vals):
                return op(*vals)

            # At least one is a sequence; normalize scalars to length-1 lists
            seq_flags = [is_seq(v) for v in vals]
            seq_vals = []
            for v, is_s in zip(vals, seq_flags):
                if is_s:
                    seq_vals.append(list(v))  # convert tuples to lists to keep storage consistent
                else:
                    seq_vals.append([v])      # scalar → length-1 for broadcasting

            # Determine target length for this axis (NumPy-style)
            lengths = [len(v) for v in seq_vals]
            non_ones = [L for L in lengths if L != 1]
            if len(non_ones) == 0:
                # All length-1 → target 1 (will recurse into next axis)
                target_len = 1
            else:
                # All non-1 must match
                if len(set(non_ones)) != 1:
                    raise ValueError(f"Incompatible lengths at broadcast axis: {lengths}")
                target_len = non_ones[0]

            # Expand each operand along this axis
            expanded = []
            for v, L in zip(seq_vals, lengths):
                if L == target_len:
                    expanded.append(v)
                elif L == 1:
                    # Repeat the single element to match target_len
                    expanded.append(v * target_len if v else [None] * target_len)
                else:
                    # Should be prevented by the check above
                    raise ValueError(f"Cannot broadcast length {L} to {target_len}")

            # Recurse elementwise over this axis
            out = []
            for items in zip(*expanded):
                out.append(recurse(*items))
            return out

        return recurse(*arrays)
  
    def _broadcast_shape(self, *shapes):
        """
        Compute the common broadcasted shape for one or more shapes, NumPy-style.
        """
        if not shapes:
            return ()
        shapes = tuple(tuple(s) for s in shapes)

        max_rank = max(len(s) for s in shapes)
        result = []

        for i in range(1, max_rank + 1):
            dims = [s[-i] if i <= len(s) else 1 for s in shapes]
            non_ones = [d for d in dims if d != 1]
            if len(set(non_ones)) > 1:
                raise ValueError(f"Shapes {shapes} are not broadcastable at axis -{i}: {dims}")
            result.append(max(dims) if dims else 1)

        return tuple(reversed(result))

    
    def _broadcast_to(self, data, from_shape, to_shape):
        """
        Expand `data` with shape `from_shape` to `to_shape` using NumPy-style rules.
        Assumes shapes are compatible (use _broadcast_shape beforehand).
        Always returns a nested Python list matching `to_shape`.
        """
        if from_shape == to_shape:
            return data

        # Scalar source: replicate before any padding
        if len(from_shape) == 0:
            def fill(shape):
                if not shape:
                    return data
                return [fill(shape[1:]) for _ in range(shape[0])]
            return fill(to_shape)

        # If from_shape is shorter, conceptually left-pad with 1s for alignment
        if len(from_shape) < len(to_shape):
            pad = (1,) * (len(to_shape) - len(from_shape))
            from_shape = pad + from_shape

        src_dim, dst_dim = from_shape[0], to_shape[0]

        if src_dim == dst_dim:
            if src_dim == 0:
                return []
            return [self._broadcast_to(d, from_shape[1:], to_shape[1:]) for d in data]

        if src_dim == 1:
            tile = data[0] if isinstance(data, (list, tuple)) and len(data) > 0 else data
            return [self._broadcast_to(tile, from_shape[1:], to_shape[1:]) for _ in range(dst_dim)]

        raise ValueError(f"Incompatible shapes during broadcasting: {from_shape} -> {to_shape}")


    def _apply_unary(self, a, op):
        """
        Recursively apply a unary op to nested data.
        Accepts scalars, lists, or tuples; always returns lists.
        """
        if isinstance(a, (list, tuple)):
            if len(a) == 0:
                return []
            return [self._apply_unary(x, op) for x in a]
        # scalar leaf
        return op(a)

    def __add__(self, other):
        """
        Elementwise addition with broadcasting.
        No graph mutations beyond creating the result tensor.
        """
        # Shapes captured locally for backward; no state written to tensors
        a_shape = self.shape
        b_shape = other.shape if isinstance(other, TensorT) else ()
        out_shape = self._broadcast_shape(a_shape, b_shape)

        # Forward: elementwise result (raw data) → wrap as TensorT
        result = self._elementwise_op(other, lambda a, b: a + b)
        out = TensorT(result, _op='add', _parent=(self, other) if isinstance(other, TensorT) else (self,))

        # Backward uses captured shapes and unbroadcasts grads
        def backward_fn(grad_out):
            grad_self = self._unbroadcast(grad_out, out_shape, a_shape)
            if isinstance(other, TensorT):
                grad_other = self._unbroadcast(grad_out, out_shape, b_shape)
                return grad_self, grad_other
            else:
                return (grad_self,)

        out.backward_fn = backward_fn
        return out

    def __radd__(self, other):
        # Delegate so scalar + tensor works the same as tensor + scalar
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, TensorT) else TensorT(other)
        out_shape = self._broadcast_shape(self.shape, other.shape)

        # forward
        a_b = self._broadcast_to(self.data,  self.shape,  out_shape)
        b_b = self._broadcast_to(other.data, other.shape, out_shape)
        result = self._apply_elementwise(a_b, b_b, lambda a, b: a * b)

        out = TensorT(result, _op='mul', _parent=(self, other))

        def backward_fn(grad_out):
            # grad_out has out_shape
            # Broadcast operands to out_shape first
            a_b = self._broadcast_to(self.data,  self.shape,  out_shape)
            b_b = self._broadcast_to(other.data, other.shape, out_shape)

            # d/dA (A*B) = B ; d/dB (A*B) = A
            grad_self_full  = self._apply_elementwise(grad_out, b_b, lambda g, b: g * b)
            grad_other_full = self._apply_elementwise(grad_out, a_b, lambda g, a: g * a)

            # Unbroadcast back to each original shape
            grad_self  = self._unbroadcast(grad_self_full,  out_shape, self.shape)
            grad_other = self._unbroadcast(grad_other_full, out_shape, other.shape)
            return grad_self, grad_other

        out.backward_fn = backward_fn
        return out


    def __rmul__(self, other):
        """
        Reverse multiplication: scalar * tensor.
        """
        return self.__mul__(other)

    def __sub__(self, other):
        """
        Elementwise subtraction with broadcasting: self - other.
        """
        a_shape = self.shape
        b_shape = other.shape if isinstance(other, TensorT) else ()
        out_shape = self._broadcast_shape(a_shape, b_shape)

        result = self._elementwise_op(other, lambda a, b: a - b)
        out = TensorT(result, _op='sub', _parent=(self, other) if isinstance(other, TensorT) else (self,))

        def backward_fn(grad_out):
            grad_self = self._unbroadcast(grad_out, out_shape, a_shape)
            if isinstance(other, TensorT):
                grad_other = self._unbroadcast(grad_out, out_shape, b_shape)
                grad_other = self._apply_unary(grad_other, lambda x: -x)  # negate
                return grad_self, grad_other
            else:
                return (grad_self,)

        out.backward_fn = backward_fn
        return out
    
    
    def __rsub__(self, other):      # other - self
        return TensorT(other).__sub__(self) if not isinstance(other, TensorT) else other.__sub__(self)

    def __rtruediv__(self, other):  # other / self
        return TensorT(other).__truediv__(self) if not isinstance(other, TensorT) else other.__truediv__(self)

    def __rpow__(self, other):      # other ** self
        return TensorT(other).__pow__(self) if not isinstance(other, TensorT) else other.__pow__(self)


    def __truediv__(self, other):
    # Compute elementwise division
        a_shape = self.shape
        b_shape = other.shape if isinstance(other, TensorT) else ()
        out_shape = self._broadcast_shape(a_shape, b_shape)

        result = self._elementwise_op(other, lambda x, y: x / y)
        out = TensorT(result, _op='div', _parent=(self, other) if isinstance(other, TensorT) else (self,))

        def backward_fn(grad_output):
            # grad w.r.t self: grad_output / other, unbroadcasted to self.shape
            grad_self = self._apply_elementwise(
                self._unbroadcast(grad_output, out_shape, a_shape),
                other.data if isinstance(other, TensorT) else other,
                lambda x, y: x / y
            )
            if isinstance(other, TensorT):
                # grad w.r.t other: -grad_output * self / (other^2), unbroadcasted to other.shape
                grad_other = self._apply_elementwise(
                    self._unbroadcast(grad_output, out_shape, b_shape),
                    self.data,
                    lambda go, s: -go * s
                )
                grad_other = self._apply_elementwise(
                    grad_other,
                    self._apply_elementwise(
                        other.data,
                        other.data,
                        lambda x, y: x * y
                    ),
                    lambda x, y: x / y
                )
                return grad_self, grad_other
            else:
                return (grad_self,)

        out.backward_fn = backward_fn
        return out
   

    def __neg__(self):
        return TensorT(self._apply_unary(self.data, lambda x: -x))
    
    def __pow__(self, other):
        """
        Elementwise power with broadcasting: self ** other.
        """
        a_shape = self.shape
        b_shape = other.shape if isinstance(other, TensorT) else ()
        out_shape = self._broadcast_shape(a_shape, b_shape)

        result = self._elementwise_op(other, lambda a, b: a ** b)
        out = TensorT(result, _op='pow', _parent=(self, other) if isinstance(other, TensorT) else (self,))

        # def backward_fn(grad_out):
        #     # full gradient logic belongs here (once), not in rpow
        #     pass  # to be filled like the others, but only here

        # out.backward_fn = backward_fn
        return out


    def tlog(self):
        """Element-wise natural logarithm."""
        return TensorT(self._apply_unary(self.data, math.log))

    def texp(self):
        """Element-wise exponential."""
        return TensorT(self._apply_unary(self.data, math.exp))
    
    def tsum(self):
        """Return the sum of all elements, any shape (list-based)."""
        def _sum(x):
            if isinstance(x, list):
                return sum(_sum(y) for y in x)
            return x
        return _sum(self.data)

    def tmean(self):
        """Return the mean (average) of all elements, any shape (list-based)."""
        total_elems = 1
        for d in self.shape:
            total_elems *= d

    def __getitem__(self, idx):
        return self.data[idx]


# DEFINING RANDOM TENSORS
    @classmethod
    def unit_tensor(cls, unit: float, shape):
        """Create a tensor filled with 0 or 1 (float)."""
        if unit not in (0, 1):
            raise ValueError("unit must be 0 or 1")
        unit = float(unit)
        def build(s):
            if len(s) == 0:
                return unit
            return [build(s[1:]) for _ in range(s[0])]
        return cls(build(shape))

    @classmethod
    def const_tensor(cls, value: float, shape):
        """Create a tensor filled with a constant value (float)."""
        value = float(value)
        def build(s):
            if len(s) == 0:
                return value
            return [build(s[1:]) for _ in range(s[0])]
        return cls(build(shape))

    @classmethod
    def random_tensor(cls, shape):
        """Create a tensor with random values in [0,1), any shape."""
        def build(s):
            if len(s) == 0:
                return random.random()
            return [build(s[1:]) for _ in range(s[0])]
        return cls(build(shape))

    def tmatmul(self, other):
        """Matrix multiplication supporting batch dimensions."""
        assert isinstance(self, TensorT) and isinstance(other, TensorT), "Both operands must be TensorT"

        # ---------- helpers ----------
        def _transpose_2d(m):
            rows = len(m); cols = len(m[0])
            return [[m[r][c] for r in range(rows)] for c in range(cols)]

        def matmul_2d(a, b):
            """Strict 2D matrix multiplication on list-of-lists."""
            if (not isinstance(a, list) or not a or not isinstance(a[0], list) or
                not isinstance(b, list) or not b or not isinstance(b[0], list)):
                raise ValueError("Both inputs must be non-empty 2D lists")
            rows_a, cols_a = len(a), len(a[0])
            rows_b, cols_b = len(b), len(b[0])
            if cols_a != rows_b:
                raise ValueError(f"Incompatible: ({rows_a}x{cols_a}) @ ({rows_b}x{cols_b})")
            out = []
            for i in range(rows_a):
                row = []
                for j in range(cols_b):
                    row.append(sum(a[i][k] * b[k][j] for k in range(cols_a)))
                out.append(row)
            return out

        def safe_int(x):
            while isinstance(x, (tuple, list)) and len(x) == 1:
                x = x[0]
            return int(x)

        def clean_shape(shape):
            if not isinstance(shape, (tuple, list)):
                return (safe_int(shape),)
            return tuple(safe_int(s) for s in shape)

        # ---------- validate ----------
        if len(self.shape) == 0 or len(other.shape) == 0:
            raise ValueError("Cannot perform matrix multiplication on scalar tensors")

        self_clean_shape = clean_shape(self.shape)
        other_clean_shape = clean_shape(other.shape)

        # ---------- coerce 1-D to 2-D for compute ----------
        if len(self_clean_shape) == 1:
            self_shape = (1, self_clean_shape[0])
            self_data_prep = [self.data]                 # row vector (1, n)
        else:
            self_shape = self_clean_shape
            self_data_prep = self.data

        if len(other_clean_shape) == 1:
            other_shape = (other_clean_shape[0], 1)      # column vector (n, 1)   <-- FIX #2
            other_data_prep = [[x] for x in other.data]
        else:
            other_shape = other_clean_shape
            other_data_prep = other.data

        # quick inner-dim check
        if self_shape[-1] != other_shape[-2]:
            raise ValueError(f"Cannot multiply: {self_shape[-1]} != {other_shape[-2]}")

        # ---------- forward ----------
        # Case 1: Pure 2-D (after coercion)
        if len(self_shape) == 2 and len(other_shape) == 2:
            result_data = matmul_2d(self_data_prep, other_data_prep)
            # Squeeze to 1-D if one original arg was 1-D  <-- FIX #3
            if len(self_clean_shape) == 1 and len(other_clean_shape) >= 2:
                result_data = result_data[0]                 # (1, n) -> (n,)
            elif len(self_clean_shape) >= 2 and len(other_clean_shape) == 1:
                result_data = [row[0] for row in result_data]  # (m, 1) -> (m,)
        else:
            # Batch cases: extract per-batch 2D slices for both sides and multiply
            def extract_from_batched(data, batch_dims):
                if not batch_dims:
                    return [data]
                mats = []
                def rec(d, dims):
                    if not dims:
                        mats.append(d)
                        return
                    for item in d:
                        rec(item, dims[1:])
                rec(data, batch_dims)
                return mats

            batch_self = self_shape[:-2] if len(self_shape) > 2 else ()
            batch_other = other_shape[:-2] if len(other_shape) > 2 else ()

            if batch_self and not batch_other:
                self_mats  = extract_from_batched(self_data_prep,  batch_self)
                other_mats = [other_data_prep] * len(self_mats)
            elif not batch_self and batch_other:
                other_mats = extract_from_batched(other_data_prep, batch_other)
                self_mats  = [self_data_prep] * len(other_mats)
            else:
                # both batched or neither (but reached here ⇒ both batched)
                self_mats  = extract_from_batched(self_data_prep,  batch_self)
                other_mats = extract_from_batched(other_data_prep, batch_other)

            # ensure each slice is 2D (handle accidental vectors)
            def ensure_2d_left(m):   # for A: vector => row
                if not isinstance(m, list):
                    return [[m]]
                if m and isinstance(m[0], (int, float)):
                    return [m]
                return m
            def ensure_2d_right(m):  # for B: vector => col
                if not isinstance(m, list):
                    return [[m]]
                if m and isinstance(m[0], (int, float)):
                    return [[x] for x in m]
                return m

            result_mats = []
            for i, (a_mat, b_mat) in enumerate(zip(self_mats, other_mats)):
                a2 = ensure_2d_left(a_mat)
                b2 = ensure_2d_right(b_mat)
                result_mats.append(matmul_2d(a2, b2))

            def reconstruct(mats, batch_dims):
                if not batch_dims:
                    return mats[0]
                it = iter(mats)
                def build(dims):
                    if not dims:
                        return next(it)
                    return [build(dims[1:]) for _ in range(dims[0])]
                return build(batch_dims)

            if batch_self:
                result_data = reconstruct(result_mats, batch_self)
            elif batch_other:
                result_data = reconstruct(result_mats, batch_other)
            else:
                result_data = result_mats  # unreachable here, but harmless

        # ---------- build output & backward ----------
        out = TensorT(result_data, _op='matmul', _parent=(self, other))

        def backward_fn(grad_op):
            """Proper backward pass for matrix multiplication with robust batch support."""

            # Simple 2D (includes coerced 1D on either side)  <-- FIX #4
            if len(self_clean_shape) <= 2 and len(other_clean_shape) <= 2:
                # grad_op is 2D if result was 2D, or 1D if result squeezed
                # Normalize grad_op to 2D for compute:
                g = grad_op
                if not isinstance(g[0], list):  # 1D -> (1, n) row
                    g = [g]

                A = self_data_prep
                B = other_data_prep

                # Transposes
                B_T = _transpose_2d(B)
                A_T = _transpose_2d(A)

                grad_self_data  = matmul_2d(g, B_T)   # shape like A
                grad_other_data = matmul_2d(A_T, g)   # shape like B

                # Squeeze grads to 1-D if original arg was 1-D
                if len(self_clean_shape) == 1:
                    grad_self_data = grad_self_data[0]          # (1, n) -> (n,)
                if len(other_clean_shape) == 1:
                    grad_other_data = [x[0] for x in grad_other_data]  # (n, 1) -> (n,)
                return grad_self_data, grad_other_data

            # ---- Batch cases (keep your existing logic but use robust transposes) ----
            def flatten_dims(dims):
                if not isinstance(dims, (tuple, list)):
                    return [dims]
                out_ = []
                for d in dims:
                    if isinstance(d, (tuple, list)):
                        out_.extend(flatten_dims(d))
                    else:
                        out_.append(int(d))
                return out_

            def extract_from_batched(data, batch_dims):
                dims = flatten_dims(batch_dims)
                if not dims:
                    return [data]
                mats = []
                def rec(d, rem):
                    if not rem:
                        mats.append(d)
                        return
                    for item in d:
                        rec(item, rem[1:])
                rec(data, dims)
                return mats

            def reconstruct_robust(mats, batch_dims):
                dims = flatten_dims(batch_dims)
                if not dims:
                    return mats[0]
                it = iter(mats)
                def build(rem):
                    if not rem:
                        return next(it)
                    return [build(rem[1:]) for _ in range(rem[0])]
                return build(dims)

            batch_self = self_shape[:-2] if len(self_shape) > 2 else ()
            batch_other = other_shape[:-2] if len(other_shape) > 2 else ()

            # prepare per-batch slices same way as forward
            if batch_self and not batch_other:
                g_mats = extract_from_batched(grad_op, batch_self)
                A_mats = extract_from_batched(self_data_prep, batch_self)
                B = other_data_prep
                B_T = _transpose_2d(B)

                grad_self_mats = []
                grad_other_sum = None
                for g, A in zip(g_mats, A_mats):
                    gs = matmul_2d(g, B_T)
                    grad_self_mats.append(gs)
                    A_T = _transpose_2d(A)
                    go = matmul_2d(A_T, g)
                    if grad_other_sum is None:
                        grad_other_sum = go
                    else:
                        # elementwise add
                        for r in range(len(go)):
                            for c in range(len(go[0])):
                                grad_other_sum[r][c] += go[r][c]
                return reconstruct_robust(grad_self_mats, batch_self), grad_other_sum

            elif not batch_self and batch_other:
                g_mats = extract_from_batched(grad_op, batch_other)
                B_mats = extract_from_batched(other_data_prep, batch_other)
                A = self_data_prep
                A_T = _transpose_2d(A)

                grad_other_mats = []
                grad_self_sum = None
                for g, B in zip(g_mats, B_mats):
                    go = matmul_2d(A_T, g)
                    grad_other_mats.append(go)
                    B_T = _transpose_2d(B)
                    gs = matmul_2d(g, B_T)
                    if grad_self_sum is None:
                        grad_self_sum = gs
                    else:
                        for r in range(len(gs)):
                            for c in range(len(gs[0])):   # <-- fixed inner bound
                                grad_self_sum[r][c] += gs[r][c]
                return grad_self_sum, reconstruct_robust(grad_other_mats, batch_other)

            else:
                # both batched
                batch_dims = batch_self or batch_other
                g_mats = extract_from_batched(grad_op, batch_dims)
                A_mats = extract_from_batched(self_data_prep, batch_self)
                B_mats = extract_from_batched(other_data_prep, batch_other)

                grad_self_mats = []
                grad_other_mats = []
                for g, A, B in zip(g_mats, A_mats, B_mats):
                    B_T = _transpose_2d(B)
                    A_T = _transpose_2d(A)
                    grad_self_mats.append(matmul_2d(g, B_T))
                    grad_other_mats.append(matmul_2d(A_T, g))

                return (reconstruct_robust(grad_self_mats, batch_self),
                        reconstruct_robust(grad_other_mats, batch_other))

        out.backward_fn = backward_fn
        return out

    def _batch_transpose(self):
        """Transpose last two dimensions for batch operations"""
        if len(self.shape) < 2:
            raise ValueError("Need at least 2 dimensions for batch transpose")
        
        def _transpose_last_two(data, shape):
            if len(shape) == 2:
                rows, cols = shape
                return [[data[r][c] for r in range(rows)] for c in range(cols)]
            else:
                return [_transpose_last_two(sub_data, shape[1:]) for sub_data in data]
        
        new_shape = self.shape[:-2] + (self.shape[-1], self.shape[-2])
        transposed_data = _transpose_last_two(self.data, self.shape)
        return TensorT(transposed_data)


    def ttranspose(self):
        """Transpose of a 2D tensor: shape (rows, cols) -> (cols, rows)."""
        if len(self.shape) != 2:
            raise ValueError("ttranspose is defined only for 2D tensors")
        rows, cols = self.shape
        transposed = [[self.data[r][c] for r in range(rows)] for c in range(cols)]
        return TensorT(transposed)


    def tflatten(self):
        """Return a flat Python list of all elements (any rank)."""
        out = []
        def rec(x):
            if isinstance(x, list):
                for y in x:
                    rec(y)
            else:
                out.append(x)
        rec(self.data)
        return out
        
  
    def treshape(self, new_shape: tuple):
        """
        Reshape to `new_shape` (any rank). Number of elements must match.
        Returns a new TensorT.
        """
        # Count elements in current and new shapes
        def numel(shape):
            n = 1
            for d in shape:
                n *= d
            return n
        old_n = numel(self.shape)
        new_n = numel(new_shape)
        if old_n != new_n:
            raise ValueError(
                f"Incompatible size for reshape: new shape {new_shape} "
                f"must have {old_n} elements"
            )

        flat = self.tflatten()
        idx = 0
        def build(shape):
            nonlocal idx
            if len(shape) == 0:
                val = flat[idx]
                idx += 1
                return val
            dim = shape[0]
            if dim == 0:
                return []
            return [build(shape[1:]) for _ in range(dim)]

        reshaped = build(tuple(new_shape))
        return TensorT(reshaped)
    
    def tsum(self):
        """Return the sum of all elements, any shape (list-based)."""
        def _sum(x):
            if isinstance(x, list):
                return sum(_sum(y) for y in x)
            return x
        return _sum(self.data)

    def tsum_axis(self, axis=0, keepdims=True):
        if len(self.shape) == 0:
            return TensorT(self.data)

        def reduce_axis(data, ax):
            if ax == 0:
                if not isinstance(data, list):
                    return data
                if len(data) == 0:
                    return []
                # list of scalars → just sum
                if not isinstance(data[0], list):
                    return sum(data)
                # otherwise transpose at this level and reduce deeper
                transposed = list(zip(*data))
                return [reduce_axis(list(group), 0) for group in transposed]
            else:
                # recurse deeper
                return [reduce_axis(d, ax - 1) for d in data]

        reduced = reduce_axis(self.data, axis)

        if keepdims:
            def add_axis(x, ax):
                if ax == 0:
                    return [x]
                # x should be a list here
                return [add_axis(sub, ax - 1) for sub in x]
            reduced = add_axis(reduced, axis)

        return TensorT(reduced)

    def tmean(self):
        """Return the mean (average) of all elements, any shape (list-based)."""
        if self.shape == ():
            return float(self.data)
        total = self.tsum()
        count = 1
        for d in self.shape:
            count *= d
        return total / count


    def tmaximum(self, scalar):
        """Element-wise maximum with scalar (for ReLU)."""
        result = self._apply_unary(self.data, lambda x: max(x, scalar))
        return TensorT(result)

    def tclip(self, min_val, max_val):
        """Clip values between min_val and max_val (for numerical stability)."""
        result = self._apply_unary(self.data, lambda x: max(min_val, min(max_val, x)))
        return TensorT(result)

    def to_list(self):
        """Return the underlying data as nested Python lists."""
        return self.data

    @classmethod
    def from_numpy(cls, np_array):
        """Convert numpy array to TensorT"""
        return cls(np_array.tolist())

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
        # 1) Normalize incoming grad
        if isinstance(grad, TensorT):
            grad = grad.data

        if grad is None:
            # If no explicit grad given, use 1 for scalars, or ones-like for tensors
            grad = 1.0 if self.shape == () else TensorT.unit_tensor(1.0, self.shape).data

        # 2) Accumulate (lazy-init if needed)
        if self.grad is None:
            self.grad = grad
        else:
            # handle scalar-vs-list safely
            if isinstance(self.grad, list) and isinstance(grad, list):
                self.grad = self._apply_elementwise(self.grad, grad, lambda x, y: x + y)
            else:
                # both scalars (or fallback)
                self.grad = (self.grad if not isinstance(self.grad, list) else self.grad)  # no-op
                self.grad = (self.grad + grad) if not isinstance(self.grad, list) else self._apply_elementwise(self.grad, grad, lambda x, y: x + y)

        # Leaf / no backward_fn
        if not self._parent or self.backward_fn is None:
            return

        # 3) Get grads for parents
        parent_grads = self.backward_fn(grad)

        # 4) Recurse only when we actually have a gradient
        for parent, parent_grad in zip(self._parent, parent_grads):
            if parent is None:
                continue
            if parent_grad is None:
                # No gradient flows to this parent (common in broadcasting); skip
                continue
            if isinstance(parent_grad, TensorT):
                parent_grad = parent_grad.data

            # Optional: if you want to be extra safe with broadcasting, unbroadcast here.
            # try:
            #     parent_grad = self._unbroadcast(parent_grad, self.shape, parent.shape)
            # except Exception:
            #     pass

            if isinstance(parent, TensorT):
                parent.backward(grad=parent_grad)
            else:
                raise ValueError("Parent must be a TensorT instance")

            