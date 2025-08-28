# Phase 1 tests: TensorT initialization, shape inference, dtype policy, rectangularity
from tensort_v2_1 import TensorT
def run_tests():
    print("=== Scalars ===")
    t1 = TensorT(5)                     # int scalar, cast to float32
    print(t1)       # shape (), dtype float32
    t2 = TensorT(5.5)                   # float scalar
    print(t2)       # shape (), dtype float32

    print("\n=== Vectors ===")
    t3 = TensorT([1, 2, 3])             # int vector → float32
    print(t3)       # shape (3,), dtype float32
    t4 = TensorT([1.0, 2.5, 3.0])       # float vector
    print(t4)       # shape (3,), dtype float32
    t5 = TensorT([True, False, True])   # bool vector → cast to 1.0,0.0,1.0 float32
    print(t5)       # shape (3,), dtype float32

    print("\n=== Matrices ===")
    t6 = TensorT([[1, 2], [3, 4]])      # matrix of ints → float32
    print(t6)       # shape (2,2), dtype float32
    t7 = TensorT([[1.2, 2.3], [3.4, 4.5]])
    print(t7)       # shape (2,2), dtype float32

    print("\n=== Higher Rank ===")
    t8 = TensorT([[[1,2],[3,4]], [[5,6],[7,8]]]) # shape (2,2,2) → float32
    print(t8)
    t9 = TensorT([[[[1],[2]], [[3],[4]]]])       # shape (1,2,2,1) → float32
    print(t9)

    print("\n=== Mixed int+float (promotion) ===")
    t10 = TensorT([1, 2.5, 3])          # mix → float32
    print(t10)

    print("\n=== Empty structures ===")
    t11 = TensorT([])                   # shape (0,)
    print(t11)
    t12 = TensorT([[], []])             # shape (2,0)
    print(t12)

    print("\n=== Ragged (should raise ValueError) ===")
    try:
        TensorT([[1,2],[3]])            # ragged 2D
    except ValueError as e:
        print("Caught ragged error as expected:", e)

    try:
        TensorT([[1,2],[3,4,5]])        # ragged 2D
    except ValueError as e:
        print("Caught ragged error as expected:", e)

    print("\n=== Non-numeric (should raise TypeError) ===")
    try:
        TensorT(["a","b"])
    except TypeError as e:
        print("Caught non-numeric error as expected:", e)

    print("\n=== Large tensors (repr truncation check) ===")
    big_vec = TensorT(list(range(20)))
    print(big_vec)                      # should show head ... tail
    big_mat = TensorT([list(range(10)) for _ in range(10)])
    print(big_mat)                      # should show top ... bottom rows

run_tests()
