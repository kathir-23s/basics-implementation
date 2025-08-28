from tensort_v2_1 import TensorT

def test_batch_matmul_gradients():
    print("===== Testing Batch Matrix Multiplication Gradients =====")
    
    # Batch case: (2,2,2) @ (2,2) 
    batch_a = TensorT([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]]
    ])
    b = TensorT([[0.5, 1.0], [1.5, 2.0]])
    
    result = batch_a.tmatmul(b)
    print("Batch forward result:", result)
    
    # Test gradients
    batch_a.zero_grad()
    b.zero_grad()
    result.backward()
    
    print("Batch gradient of a:", TensorT(batch_a.grad) if batch_a.grad else "None")
    print("Gradient of b:", TensorT(b.grad) if b.grad else "None")

# Run this test to verify batch gradients work too

if __name__ == "__main__":
    # test_matmul_gradients()
    test_batch_matmul_gradients()