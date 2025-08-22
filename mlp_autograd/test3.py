# bench_matmul_compare_big.py
# Compare: TensorT.tmatmul (naive) vs TensorT.block_matmul vs matmul_fast_pure (pure-Python)
# Uses only your TensorT API. No external deps.

import time
import random
import argparse
from tensor.tensor_scratch import TensorT

# ---------- pure-Python "fast" kernel (cache-friendlier) ----------
def matmul_fast_pure(A: TensorT, B: TensorT) -> TensorT:
    """
    Pure-Python cache-friendlier matmul.
    (m x k) @ (k x n) -> (m x n)
    Strategy: materialize columns of B once so inner loop is tight over Python lists.
    """
    assert isinstance(A, TensorT) and isinstance(B, TensorT), "Both must be TensorT"
    assert len(A.shape) == 2 and len(B.shape) == 2, "Both tensors must be 2D"
    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        raise ValueError(f"Incompatible shapes: {A.shape} @ {B.shape}")

    # Use your transpose (returns TensorT), take raw lists to avoid extra object overhead
    B_T_cols = B.ttranspose().data   # list-of-rows, each row is a column of original B

    C = [[0.0 for _ in range(n)] for _ in range(m)]
    Ad = A.data
    for i in range(m):
        Ai = Ad[i]        # row i of A
        Ci = C[i]
        # dot(Ai, B[:, j]) using B_T_cols[j]
        for j in range(n):
            bj = B_T_cols[j]
            s = 0.0
            # tight inner loop over k
            for kk in range(k):
                s += Ai[kk] * bj[kk]
            Ci[j] = s

    out = TensorT(C, _op='matmul_fast_pure', _parent=(A, B))
    def backward_fn(grad_op):
        # dA = G @ B^T ; dB = A^T @ G using the same fast kernel
        dA = matmul_fast_pure(TensorT(grad_op), B.ttranspose()).data
        dB = matmul_fast_pure(A.ttranspose(), TensorT(grad_op)).data
        return dA, dB
    out.backward_fn = backward_fn
    return out

# ---------- helpers ----------
def time_once(fn):
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return (t1 - t0), out

def max_abs_diff_list(a, b):
    # nested list max |a-b|
    if isinstance(a, list) and isinstance(b, list):
        if not a and not b:
            return 0.0
        if isinstance(a[0], list):
            return max(max_abs_diff_list(x, y) for x, y in zip(a, b))
        else:
            return max(abs(x - y) for x, y in zip(a, b))
    raise TypeError("Expected nested lists")

def allclose_list(a, b, atol=1e-9):
    # nested list allclose
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): 
            return False
        if not a:
            return True
        if isinstance(a[0], list):
            return all(allclose_list(x, y, atol) for x, y in zip(a, b))
        else:
            return all(abs(x - y) <= atol for x, y in zip(a, b))
    return False

def bench_case(A_shape, B_shape, block_size=16, reps=3, atol=1e-9, label="case"):
    print(f"\n=== {label} ===")
    print(f"A @ B  with shapes {A_shape} @ {B_shape}")
    random.seed(0)  # reproducible
    A = TensorT.random_tensor(A_shape)
    B = TensorT.random_tensor(B_shape)

    # warmups
    _ = A.tmatmul(B)
    _ = A.block_matmul(B if A_shape[1]==B_shape[0] and A_shape[0]==A_shape[1] and B_shape[0]==B_shape[1] else B, block_size) \
        if (A_shape[0]==A_shape[1]==B_shape[0]==B_shape[1]) else None
    _ = matmul_fast_pure(A, B)

    # timings
    t_naive = 0.0
    t_block = 0.0
    t_fast  = 0.0
    C_naive = C_block = C_fast = None

    for _ in range(reps):
        dt, out = time_once(lambda: A.tmatmul(B))
        t_naive += dt
        C_naive = out

        if A_shape[0]==A_shape[1]==B_shape[0]==B_shape[1]:
            dt, out = time_once(lambda: A.block_matmul(B, block_size))
            t_block += dt
            C_block = out
        else:
            t_block += 0.0
            C_block = None

        dt, out = time_once(lambda: matmul_fast_pure(A, B))
        t_fast += dt
        C_fast = out

    print(f"avg over {reps} reps:")
    print(f"  tmatmul       : {t_naive/reps:.6f} s")

    if C_block is not None:
        print(f"  block_matmul  : {t_block/reps:.6f} s   (block={block_size})")
    else:
        print(f"  block_matmul  : n/a (only defined for equal-size square matrices)")

    print(f"  matmul_fast   : {t_fast/reps:.6f} s   (pure-Python)")

    # correctness checks
    print("value checks (max |Δ|):")
    if C_block is not None:
        d_nb = max_abs_diff_list(C_naive.data, C_block.data)
        ok_nb = allclose_list(C_naive.data, C_block.data, atol=atol)
        print(f"  naive vs block : {d_nb:.3e}   allclose={ok_nb}")
    else:
        print("  naive vs block : n/a")

    d_nf = max_abs_diff_list(C_naive.data, C_fast.data)
    ok_nf = allclose_list(C_naive.data, C_fast.data,  atol=atol)
    print(f"  naive vs fast  : {d_nf:.3e}   allclose={ok_nf}")

    if C_block is not None:
        d_bf = max_abs_diff_list(C_block.data, C_fast.data)
        ok_bf = allclose_list(C_block.data, C_fast.data,  atol=atol)
        print(f"  block vs fast  : {d_bf:.3e}   allclose={ok_bf}")
    else:
        print("  block vs fast  : n/a")

# ---------- suites ----------
def suite_quick(block, reps, atol):
    # Square
    bench_case((256, 256), (256, 256), block, reps, atol, label="square 256")
    bench_case((512, 512), (512, 512), block, reps, atol, label="square 512")
    # MLP-like (your logs)
    bench_case((32, 8),   (8, 16512),  block, reps, atol, label="MLP: 32x8 @ 8x16512")
    bench_case((16, 32),  (32, 16512), block, reps, atol, label="MLP: 16x32 @ 32x16512")
    bench_case((1, 16),   (16, 16512), block, reps, atol, label="MLP: 1x16 @ 16x16512")

def suite_medium(block, reps, atol):
    # Bigger squares
    bench_case((768, 768), (768, 768), block, reps, atol, label="square 768")
    bench_case((1024,1024),(1024,1024),block, reps, atol, label="square 1024")
    # Wider/taller rectangulars (keep total ops manageable)
    bench_case((64, 64),   (64, 8192),  block, reps, atol, label="rect: 64x64 @ 64x8192")
    bench_case((128, 64),  (64, 4096),  block, reps, atol, label="rect: 128x64 @ 64x4096")
    bench_case((64, 128),  (128, 4096), block, reps, atol, label="rect: 64x128 @ 128x4096")

def suite_heavy(block, reps, atol):
    # Hefty squares — WARNING: pure Python can be slow
    bench_case((1280,1280),(1280,1280),block, reps, atol, label="square 1280")
    # You can try these one by one if you want to push harder:
    # bench_case((1536,1536),(1536,1536),block, reps, atol, label="square 1536")
    # bench_case((1792,1792),(1792,1792),block, reps, atol, label="square 1792")
    # Large skinny/wide (moderated to avoid billions of ops)
    bench_case((128,128),  (128, 6144), block, reps, atol, label="rect: 128x128 @ 128x6144")
    bench_case((256,64),   (64, 4096),  block, reps, atol, label="rect: 256x64  @ 64x4096")
    bench_case((64,256),   (256,2048),  block, reps, atol, label="rect: 64x256  @ 256x2048")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["quick","medium","heavy","all"], default="quick",
                    help="which size suite to run")
    ap.add_argument("--block", type=int, default=16, help="block size for block_matmul")
    ap.add_argument("--reps", type=int, default=3, help="repetitions per case")
    ap.add_argument("--atol", type=float, default=1e-9, help="allclose tolerance")
    args = ap.parse_args()

    print("\n--- Matmul Bench (pure Python) ---")
    print(f"suite={args.suite}, block={args.block}, reps={args.reps}, atol={args.atol}")

    if args.suite in ("quick","all"):
        suite_quick(args.block, args.reps, args.atol)
    if args.suite in ("medium","all"):
        suite_medium(args.block, args.reps, args.atol)
    if args.suite in ("heavy","all"):
        suite_heavy(args.block, args.reps, args.atol)

if __name__ == "__main__":
    main()
