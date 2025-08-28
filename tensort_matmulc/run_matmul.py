import argparse, random, time, os, sys, gc

# Ensure project root is on sys.path (so mm_backend / tensor_scratch import works)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from mm_backend import set_backend, PY_NAIVE, C_BACKEND
from matmul_c_tensorT import TensorT


def rand_mat(rows, cols, rng):
    return [[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def time_once(A, B):
    t0 = time.perf_counter()
    C = A.tmatmul(B)
    t1 = time.perf_counter()
    # Use the result a tiny bit so nothing gets optimized away (paranoia).
    _ = C.shape
    return t1 - t0


def main():
    p = argparse.ArgumentParser(description="Time matmul with selected backend.")
    p.add_argument("--backend", choices=["py_naive", "c"], default="c",
                   help="Which backend to use.")
    p.add_argument("--m", type=int, default=128, help="Rows of A")
    p.add_argument("--k", type=int, default=128, help="Cols of A / Rows of B")
    p.add_argument("--n", type=int, default=128, help="Cols of B")
    p.add_argument("--reps", type=int, default=3, help="Timed repetitions; best (min) is reported")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs not counted")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    # Optional: control OpenMP threads if you built the C lib with -fopenmp
    p.add_argument("--omp_threads", type=int, default=None,
                   help="Set OMP_NUM_THREADS for C backend (set before first call).")
    args = p.parse_args()

    # Backend select
    if args.backend == "py_naive":
        set_backend(PY_NAIVE)
    else:
        if args.omp_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
        set_backend(C_BACKEND)

    rng = random.Random(args.seed)

    # Create matrices once (so we only time the multiply)
    A = TensorT(rand_mat(args.m, args.k, rng))
    B = TensorT(rand_mat(args.k, args.n, rng))

    # Warmup
    for _ in range(max(0, args.warmup)):
        _ = A.tmatmul(B)
    gc.collect()

    # Timed reps
    times = []
    for _ in range(max(1, args.reps)):
        gc.collect()
        times.append(time_once(A, B))

    # Report only timings
    print(f"backend={args.backend}  shape=({args.m}x{args.k})*({args.k}x{args.n})")
    print(f"min_time_s={min(times):.6f}  avg_time_s={sum(times)/len(times):.6f}  reps={len(times)}  warmup={args.warmup}")

if __name__ == "__main__":
    main()