
from tensor.tensor_scratch import TensorT

def _print_matrix(name, M, r=2, c=4):
    # show just a small top-left corner to keep it readable
    print(f"{name} shape={len(M)}x{len(M[0]) if M else 0}")
    for i in range(min(r, len(M))):
        print("  ", [round(M[i][j], 4) for j in range(min(c, len(M[0])))])

def test_block_matmul_prints():
    # Rectangular: (5x3) @ (3x7) -> (5x7)
    A = TensorT.random_tensor((5, 3))
    B = TensorT.random_tensor((3, 7))
    C_blk = A.block_matmul(B, block_size=2)
    C_ref = A.tmatmul(B)

    same = (C_blk.data == C_ref.data)
    print("Rectangular (5x3)@(3x7):", "PASS" if same else "FAIL")
    _print_matrix("C_blk", C_blk.data)
    _print_matrix("C_ref", C_ref.data)

    # Square: (6x6) @ (6x6)
    A2 = TensorT.random_tensor((6, 6))
    B2 = TensorT.random_tensor((6, 6))
    C2_blk = A2.block_matmul(B2, block_size=3)
    C2_ref = A2.tmatmul(B2)

    same2 = (C2_blk.data == C2_ref.data)
    print("Square (6x6)@(6x6):", "PASS" if same2 else "FAIL")
    _print_matrix("C2_blk", C2_blk.data)
    _print_matrix("C2_ref", C2_ref.data)

test_block_matmul_prints()
