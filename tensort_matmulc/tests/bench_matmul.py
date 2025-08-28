import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# tests/bench_matmul.py
import time, random
from mm_backend import use_backend, PY_NAIVE, C_BACKEND
from matmul_c_tensorT import TensorT

def rand_mat(r, c):
    return [[random.random() for _ in range(c)] for _ in range(r)]

def time_once(fn):
    t0 = time.perf_counter(); fn(); t1 = time.perf_counter(); return t1 - t0

def bench_pair(shapeA, shapeB, reps=3):
    A = TensorT(rand_mat(*shapeA))
    B = TensorT(rand_mat(*shapeB))

    with use_backend(PY_NAIVE):
        t_py = min(time_once(lambda: A.tmatmul(B)) for _ in range(reps))
    with use_backend(C_BACKEND):
        t_c  = min(time_once(lambda: A.tmatmul(B)) for _ in range(reps))

    spd = t_py / max(t_c, 1e-12)
    print(f"{shapeA} x {shapeB}  ->  py:{t_py:.6f}s  c:{t_c:.6f}s  speedup×{spd:.2f}")

if __name__ == "__main__":
    random.seed(0)
    # Classic square
    # bench_pair((256,256), (256,256), reps=10)
    # bench_pair((384,384), (384,384), reps=10)
    # Your MLP-ish shapes
    bench_pair((128,784), (784,5000), reps=1)
    # bench_pair((64,10), (10,6000), reps=2)     # try a tall skinny multiply
