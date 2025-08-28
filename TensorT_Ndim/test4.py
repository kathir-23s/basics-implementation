from tensort_v2_1 import TensorT

def test_matmul_gradients():
    print("===== Testing Matrix Multiplication Gradients =====")
    
    # Simple 2D case
    a = TensorT([[1.0, 2.0], [3.0, 4.0]])
    b = TensorT([[0.5, 1.0], [1.5, 2.0]])
    
    c = a.tmatmul(b)
    print("Forward result:", c)
    
    # Clear gradients and run backward
    a.zero_grad()
    b.zero_grad()
    c.backward()
    
    print("Gradient of a:", TensorT(a.grad) if a.grad else "None")
    print("Gradient of b:", TensorT(b.grad) if b.grad else "None")

if __name__ == "__main__":
    test_matmul_gradients()
