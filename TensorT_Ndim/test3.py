# Assumes TensorT class and the updated tmatmul implementation are defined and imported
from tensort_v2_1 import TensorT

def test_matmul():
    print("===== Simple 2D Matrix Multiply =====")
    a = TensorT([[1.0, 2.0], [3.0, 4.0]])
    b = TensorT([[5.0, 6.0], [7.0, 8.0]])
    try:
        result = a.tmatmul(b)
        print("Result:\n", result)
    except Exception as e:
        print("Error in 2D matmul:", e)

    print("\n===== Batch Matrix Multiply (3D) =====")
    batch_a = TensorT([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]]
    ])  # shape (2, 2, 2)
    batch_b = TensorT([
        [[1.0, 0.0], [0.0, 1.0]],
        [[2.0, 3.0], [4.0, 5.0]]
    ])  # shape (2, 2, 2)

    try:
        batch_result = batch_a.tmatmul(batch_b)
        print("Batch result:\n", batch_result)
    except Exception as e:
        print("Error in batch matmul:", e)

    print("\n===== Broadcasting Batch Matmul =====")
    batch_a2 = TensorT([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]]
    ])  # shape (2, 2, 2)
    b2 = TensorT([[1.0, 0.0], [0.0, 1.0]])  # shape (2, 2) -- will broadcast

    try:
        broadcast_batch_result = batch_a2.tmatmul(b2)
        print("Broadcasted batch result:\n", broadcast_batch_result)
    except Exception as e:
        print("Error in broadcasting batch matmul:", e)

if __name__ == "__main__":
    test_matmul()
