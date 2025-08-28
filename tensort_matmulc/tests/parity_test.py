import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# tests/parity_test.py
import random
from mm_backend import use_backend, PY_NAIVE, C_BACKEND
from matmul_c_tensorT import TensorT

def rand_mat(r, c):
    return [[random.uniform(-1, 1) for _ in range(c)] for _ in range(r)]

def allclose_list(a, b, atol=1e-8):
    if isinstance(a, list) and isinstance(b, list):
        if not a and not b: return True
        if isinstance(a[0], list):
            return all(allclose_list(x, y, atol) for x, y in zip(a, b))
        return all(abs(x - y) <= atol for x, y in zip(a, b))
    return False

if __name__ == "__main__":
    random.seed(0)
    A = TensorT(rand_mat(5, 7))
    B = TensorT(rand_mat(7, 4))

    # Forward parity
    with use_backend(PY_NAIVE):
        C_ref = A.tmatmul(B)
    with use_backend(C_BACKEND):
        C_c = A.tmatmul(B)
    assert allclose_list(C_ref.data, C_c.data, 1e-8), "Forward mismatch"
    print("✅ forward parity OK")

    # Backward parity with a simple scalar loss: sum(C)
    def back_once(backend):
        with use_backend(backend):
            C = A.tmatmul(B)
            grad_out = [[1.0 for _ in range(C.shape[1])] for _ in range(C.shape[0])]
            dA, dB = C.backward_fn(grad_out)   # uses your stored backward_fn
            return dA, dB

    dA_py, dB_py = back_once(PY_NAIVE)
    dA_c,  dB_c  = back_once(C_BACKEND)

    assert allclose_list(dA_py, dA_c, 1e-8), "Grad dA mismatch"
    assert allclose_list(dB_py, dB_c, 1e-8), "Grad dB mismatch"
    print("✅ backward parity OK")
