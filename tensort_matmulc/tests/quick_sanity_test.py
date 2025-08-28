# tests/quick_sanity_test.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mm_backend import set_backend, C_BACKEND, PY_NAIVE
from matmul_c_tensorT import TensorT


def run_once(backend):
    set_backend(backend)
    A = TensorT([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]])          # (2x3)
    B = TensorT([[7.0,  8.0],
                 [9.0, 10.0],
                 [11.0,12.0]])               # (3x2)
    C = A.tmatmul(B)
    return C.data

if __name__ == "__main__":
    cref = run_once(PY_NAIVE)
    cfast = run_once(C_BACKEND)
    print("PY_NAIVE:", cref)   # expect [[58, 64],[139, 154]]
    print("C_BACKEND:", cfast)
    assert cref == cfast, "Mismatch between Python and C backends"
    print("✅ quick_sanity: forward parity OK")
