# fastmatmul.py
import os
import ctypes
from ctypes import c_int, c_double, POINTER

# Adjust path if needed (e.g., absolute path to your .so)
_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libfastmatmul2.so"))

# void matmul(const double* A, const double* B, double* C, int m, int k, int n)
_lib.matmul.argtypes = [
    POINTER(c_double), POINTER(c_double), POINTER(c_double),
    c_int, c_int, c_int
]
_lib.matmul.restype = None

def _flatten_row_major(mat, rows, cols):
    """mat: list[list[float]] with shape (rows, cols)"""
    buf = (c_double * (rows * cols))()
    idx = 0
    for i in range(rows):
        row = mat[i]
        # (Optional) sanity check length
        # if len(row) != cols: raise ValueError("Row length mismatch")
        for j in range(cols):
            buf[idx] = row[j]
            idx += 1
    return buf

def _to_2d_list(buf, rows, cols):
    out = []
    idx = 0
    for i in range(rows):
        row = [0.0] * cols
        for j in range(cols):
            row[j] = buf[idx]
            idx += 1
        out.append(row)
    return out

def matmul_ll(A_ll, B_ll):
    """
    A_ll: list[list[float]] shape (m,k)
    B_ll: list[list[float]] shape (k,n)
    returns: list[list[float]] shape (m,n)
    """
    m = len(A_ll);    k = len(A_ll[0]) if m else 0
    k2 = len(B_ll);   n = len(B_ll[0]) if k2 else 0
    if k != k2:
        raise ValueError("Incompatible shapes: A(m×k) and B(k×n)")

    A_buf = _flatten_row_major(A_ll, m, k)
    B_buf = _flatten_row_major(B_ll, k, n)
    C_buf = (c_double * (m * n))()

    _lib.matmul(A_buf, B_buf, C_buf, m, k, n)
    return _to_2d_list(C_buf, m, n)
