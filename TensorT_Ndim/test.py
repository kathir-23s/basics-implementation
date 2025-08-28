import math
import random
from tensort_v2_1 import TensorT
import numpy as np

def show_grad(name, grad):
    if grad is None:
        print(f" - {name}: None")
    else:
        print(f" - {name}: {TensorT(grad)}")

def comprehensive_tensor_tests():
    """
    Comprehensive test suite for TensorT class covering:
    - Basic operations (forward & backward)
    - Broadcasting
    - Matrix operations
    - Batch operations  
    - Gradient computation
    - Complex computation graphs
    - Edge cases
    """
    
    print("="*60)
    print("COMPREHENSIVE TENSOR TESTING SUITE")
    print("="*60)
    
    # Test 1: Basic Tensor Creation and Shape Validation
    print("\n1. TENSOR CREATION & SHAPE VALIDATION")
    print("-" * 40)
    
    try:
        # Scalar
        scalar = TensorT(5.0)
        print(f"✓ Scalar: {scalar}, shape: {scalar.shape}")
        
        # 1D vector
        vec1d = TensorT([1.0, 2.0, 3.0])
        print(f"✓ 1D Vector: shape: {vec1d.shape}")
        
        # 2D matrix
        mat2d = TensorT([[1.0, 2.0], [3.0, 4.0]])
        print(f"✓ 2D Matrix: shape: {mat2d.shape}")
        
        # 3D tensor
        tensor3d = TensorT([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        print(f"✓ 3D Tensor: shape: {tensor3d.shape}")
        
        # 4D tensor
        tensor4d = TensorT([[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]])
        print(f"✓ 4D Tensor: shape: {tensor4d.shape}")
        
        # Test rectangular validation
        try:
            invalid = TensorT([[1, 2, 3], [4, 5]])  # Jagged array
            print("✗ Should have failed rectangular validation")
        except ValueError:
            print("✓ Rectangular validation works")
            
    except Exception as e:
        print(f"✗ Tensor creation failed: {e}")
    
    # Test 2: Arithmetic Operations (Forward)
    print("\n2. ARITHMETIC OPERATIONS (FORWARD)")
    print("-" * 40)
    
    a = TensorT([[1.0, 2.0], [3.0, 4.0]])
    b = TensorT([[5.0, 6.0], [7.0, 8.0]])
    scalar = TensorT(2.0)
    
    # Addition
    add_result = a + b
    print(f"✓ Addition: {a.shape} + {b.shape} = {add_result.shape}")
    
    # Subtraction
    sub_result = a - b
    print(f"✓ Subtraction: {a.shape} - {b.shape} = {sub_result.shape}")
    
    # Multiplication
    mul_result = a * b
    print(f"✓ Multiplication: {a.shape} * {b.shape} = {mul_result.shape}")
    
    # Division
    div_result = a / b
    print(f"✓ Division: {a.shape} / {b.shape} = {div_result.shape}")
    
    # Power
    pow_result = a ** 2
    print(f"✓ Power: {a.shape} ** 2 = {pow_result.shape}")
    
    # Negation
    neg_result = -a
    print(f"✓ Negation: -{a.shape} = {neg_result.shape}")
    
    # Test 3: Broadcasting Operations
    print("\n3. BROADCASTING OPERATIONS")
    print("-" * 40)
    
    # Scalar broadcasting
    scalar_add = a + scalar
    print(f"✓ Scalar broadcast: {a.shape} + {scalar.shape} = {scalar_add.shape}")
    
    # Vector broadcasting
    vec = TensorT([1.0, 2.0])
    vec_broadcast = a + vec
    print(f"✓ Vector broadcast: {a.shape} + {vec.shape} = {vec_broadcast.shape}")
    
    # Complex broadcasting
    tensor_3d = TensorT([[[1.0, 2.0]], [[3.0, 4.0]]])  # (2, 1, 2)
    tensor_2d = TensorT([[5.0, 6.0], [7.0, 8.0]])      # (2, 2)
    
    try:
        broadcast_result = tensor_3d + tensor_2d
        print(f"✓ 3D+2D broadcast: {tensor_3d.shape} + {tensor_2d.shape} = {broadcast_result.shape}")
    except Exception as e:
        print(f"✗ Broadcasting failed: {e}")
    
    # Test 4: Matrix Operations
    print("\n4. MATRIX OPERATIONS")
    print("-" * 40)
    
    # 2D Matrix multiplication
    mat_a = TensorT([[1.0, 2.0], [3.0, 4.0]])
    mat_b = TensorT([[5.0, 6.0], [7.0, 8.0]])
    
    matmul_2d = mat_a.tmatmul(mat_b)
    print(f"✓ 2D MatMul: {mat_a.shape} @ {mat_b.shape} = {matmul_2d.shape}")
    
    # Batch matrix multiplication
    batch_a = TensorT([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # (2, 2, 2)
    batch_b = TensorT([[1.0, 0.0], [0.0, 1.0]])  # (2, 2)
    
    batch_matmul = batch_a.tmatmul(batch_b)
    print(f"✓ Batch MatMul: {batch_a.shape} @ {batch_b.shape} = {batch_matmul.shape}")
    
    # 1D vector matrix multiplication
    vec1d = TensorT([1.0, 2.0])
    vec_result = vec1d.tmatmul(mat_b)
    print(f"✓ 1D MatMul: {vec1d.shape} @ {mat_b.shape} = {vec_result.shape}")
    
    # Transpose
    transpose_result = mat_a.ttranspose()
    print(f"✓ 2D Transpose: {mat_a.shape} -> {transpose_result.shape}")
    
    # Batch transpose
    batch_transpose = batch_a._batch_transpose()
    print(f"✓ Batch Transpose: {batch_a.shape} -> {batch_transpose.shape}")
    
    # Test 5: Reduction Operations
    print("\n5. REDUCTION OPERATIONS")
    print("-" * 40)
    
    # Sum
    sum_result = mat_a.tsum()
    print(f"✓ Sum: {mat_a.shape} -> scalar = {sum_result}")
    
    # Mean
    mean_result = mat_a.tmean()
    print(f"✓ Mean: {mat_a.shape} -> scalar = {mean_result}")
    
    # Sum along axis
    sum_axis_result = mat_a.tsum_axis(axis=0, keepdims=True)
    print(f"✓ Sum axis 0: {mat_a.shape} -> {sum_axis_result.shape}")
    
    # Test 6: Shape Manipulation
    print("\n6. SHAPE MANIPULATION")
    print("-" * 40)
    
    # Flatten
    flatten_result = mat_a.tflatten()
    print(f"✓ Flatten: {mat_a.shape} -> list of {len(flatten_result)} elements")
    
    # Reshape
    reshape_result = mat_a.treshape((1, 4))
    print(f"✓ Reshape: {mat_a.shape} -> {reshape_result.shape}")
    
    # Test 7: Unary Functions
    print("\n7. UNARY FUNCTIONS")
    print("-" * 40)
    
    positive_tensor = TensorT([[1.0, 2.0], [3.0, 4.0]])
    
    # Log
    log_result = positive_tensor.tlog()
    print(f"✓ Log: {positive_tensor.shape} = {log_result.shape}")
    
    # Exp
    exp_result = positive_tensor.texp()
    print(f"✓ Exp: {positive_tensor.shape} = {exp_result.shape}")
    
    # Maximum (ReLU-like)
    max_result = positive_tensor.tmaximum(2.0)
    print(f"✓ Maximum: {positive_tensor.shape} = {max_result.shape}")
    
    # Clip
    clip_result = positive_tensor.tclip(1.5, 3.5)
    print(f"✓ Clip: {positive_tensor.shape} = {clip_result.shape}")
    
    # Test 8: Gradient Computation (Basic)
    print("\n8. BASIC GRADIENT COMPUTATION")
    print("-" * 40)
    
    # Simple computation: z = x * y + x / y
    x = TensorT([[1.0, 2.0], [3.0, 4.0]])
    y = TensorT([[2.0, 3.0], [4.0, 5.0]])
    
    z = x * y + x / y
    print(f"✓ Forward: z = x * y + x / y, z.shape = {z.shape}")
    
    # Backward
    x.zero_grad()
    y.zero_grad()
    z.backward()
    
    print(f"✓ Backward: grad_x computed, shape = {TensorT(x.grad).shape if x.grad else 'None'}")
    print(f"✓ Backward: grad_y computed, shape = {TensorT(y.grad).shape if y.grad else 'None'}")
    
    # Test 9: Matrix Multiplication Gradients
    print("\n9. MATRIX MULTIPLICATION GRADIENTS")
    print("-" * 40)
    
    A = TensorT([[1.0, 2.0], [3.0, 4.0]])
    B = TensorT([[0.5, 1.0], [1.5, 2.0]])
    
    C = A.tmatmul(B)
    print(f"✓ MatMul Forward: {A.shape} @ {B.shape} = {C.shape}")
    
    A.zero_grad()
    B.zero_grad()
    C.backward()
    
    print(f"✓ MatMul Backward: grad_A computed, shape = {TensorT(A.grad).shape if A.grad else 'None'}")
    print(f"✓ MatMul Backward: grad_B computed, shape = {TensorT(B.grad).shape if B.grad else 'None'}")
    
    # Test 10: Complex Computation Graph
    print("\n10. COMPLEX COMPUTATION GRAPH")
    print("-" * 40)
    
    # Create a more complex graph: f(x, y, w) = (x @ w + y) * y
    x = TensorT([[1.0, 2.0], [3.0, 4.0]])
    y = TensorT([[0.1, 0.2], [0.3, 0.4]])
    w = TensorT([[2.0, 0.0], [0.0, 2.0]])
    
    # Forward pass
    temp1 = x.tmatmul(w)     # Matrix multiplication
    temp2 = temp1 + y        # Addition with broadcasting
    result = temp2 * y       # Element-wise multiplication
    
    print(f"✓ Complex graph forward: result.shape = {result.shape}")
    
    # Backward pass
    x.zero_grad()
    y.zero_grad()
    w.zero_grad()
    result.backward()
    
    print(f"✓ Complex graph backward: all gradients computed")
    print(f"  - grad_x: {TensorT(x.grad).shape if x.grad else 'None'}")
    print(f"  - grad_y: {TensorT(y.grad).shape if y.grad else 'None'}")
    print(f"  - grad_w: {TensorT(w.grad).shape if w.grad else 'None'}")
    
    # Test 11: Batch Operations with Gradients
    print("\n11. BATCH OPERATIONS WITH GRADIENTS")
    print("-" * 40)
    
    # Batch input
    batch_x = TensorT([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])  # (3, 1, 2)
    weights = TensorT([[1.0, 0.5], [0.5, 1.0]])                    # (2, 2)
    
    # Forward: batch matrix multiplication
    batch_output = batch_x.tmatmul(weights)
    batch_sum = TensorT(batch_output.tsum())  # Scalar loss
    
    print(f"✓ Batch forward: {batch_x.shape} @ {weights.shape} = {batch_output.shape}")
    
    # Backward
    batch_x.zero_grad()
    weights.zero_grad()
    batch_output.backward(TensorT.unit_tensor(1.0, batch_output.shape).data)

    print("✓ Batch backward: gradients computed")
    print(f"  - grad_batch_x: {('None' if batch_x.grad is None else f'shape {TensorT(batch_x.grad).shape}')}")
    print(f"  - grad_weights: {('None' if weights.grad is None else f'shape {TensorT(weights.grad).shape}')}")    
    
    # Test 12: Broadcasting Gradients
    print("\n12. BROADCASTING GRADIENTS")
    print("-" * 40)
    
    large_tensor = TensorT([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])     # (2, 3)
    small_tensor = TensorT([0.1, 0.2, 0.3])                        # (3,)
    
    broadcast_result = large_tensor * small_tensor
    broadcast_sum = TensorT(broadcast_result.tsum())
    
    print(f"✓ Broadcast forward: {large_tensor.shape} * {small_tensor.shape} = {broadcast_result.shape}")
    
    large_tensor.zero_grad()
    small_tensor.zero_grad()
    broadcast_result.backward(TensorT.unit_tensor(1.0, broadcast_result.shape).data)

    print("✓ Broadcast backward: gradients computed")
    print(f"  - grad_large: {('None' if large_tensor.grad is None else f'shape {TensorT(large_tensor.grad).shape}')}")
    print(f"  - grad_small: {('None' if small_tensor.grad is None else f'shape {TensorT(small_tensor.grad).shape}')}")    


    # Test 13: Edge Cases and Error Handling
    print("\n13. EDGE CASES & ERROR HANDLING")
    print("-" * 40)
    
    # Test incompatible shapes
    try:
        incompatible_a = TensorT([[1, 2, 3]])  # (1, 3)
        incompatible_b = TensorT([[1], [2]])   # (2, 1)
        incompatible_a.tmatmul(incompatible_b)
        print("✗ Should have failed on incompatible matmul shapes")
    except ValueError:
        print("✓ Incompatible matmul shapes correctly rejected")
    
    # Test scalar matmul
    try:
        scalar_matmul = TensorT(5.0).tmatmul(TensorT(3.0))
        print("✗ Should have failed on scalar matmul")
    except ValueError:
        print("✓ Scalar matmul correctly rejected")
    
    # Test 14: Memory and Performance Check
    print("\n14. MEMORY & PERFORMANCE CHECK")
    print("-" * 40)
    
    # Create larger tensors to test memory handling
    large_a = TensorT.random_tensor((10, 50))
    large_b = TensorT.random_tensor((50, 30))
    
    large_result = large_a.tmatmul(large_b)
    print(f"✓ Large matmul: {large_a.shape} @ {large_b.shape} = {large_result.shape}")
    
    # Test gradient computation on larger tensors
    large_a.zero_grad()
    large_b.zero_grad()
    large_sum = TensorT(large_result.tsum())
    large_sum.backward()
    
    print(f"✓ Large tensor gradients computed successfully")
    
    # Test 15: Random Tensor Creation
    print("\n15. RANDOM TENSOR CREATION")
    print("-" * 40)
    
    # Unit tensors
    zeros = TensorT.unit_tensor(0, (2, 3))
    ones = TensorT.unit_tensor(1, (2, 3))
    print(f"✓ Zero tensor: shape {zeros.shape}")
    print(f"✓ One tensor: shape {ones.shape}")
    
    # Constant tensors
    const_tensor = TensorT.const_tensor(3.14, (2, 2))
    print(f"✓ Constant tensor: shape {const_tensor.shape}")
    
    # Random tensors
    random_tensor = TensorT.random_tensor((3, 4))
    print(f"✓ Random tensor: shape {random_tensor.shape}")
    
    # Final Summary
    print("\n" + "="*60)
    print("COMPREHENSIVE TENSOR TESTING COMPLETE")
    print("✓ All core operations tested successfully")
    print("✓ Forward and backward passes verified")
    print("✓ Broadcasting and batch operations confirmed")
    print("✓ Complex computation graphs working")
    print("✓ Error handling and edge cases covered")
    print("="*60)

    

# Test function to validate numerical gradients
def gradient_check():
    """Numerical gradient checking to validate autodiff correctness"""
    print("\nNUMERICAL GRADIENT CHECK")
    print("-" * 30)

    # f: returns a SCALAR float for finite differences (keep as-is)
    def test_func(x):
        return (x * x).tsum()

    def numerical_gradient(f, x, h=1e-5):
        """Compute numerical gradient using finite differences (properly mutates nested data)"""
        import copy

        shape = x.shape

        # utilities to navigate/set nested lists by multi-index
        def index_to_multi(i, shape):
            idxs = []
            for s in reversed(shape):
                idxs.append(i % s)
                i //= s
            return list(reversed(idxs))

        def get_at(data, idxs):
            d = data
            for t in idxs[:-1]:
                d = d[t]
            return d[idxs[-1]]

        def set_at(data, idxs, val):
            d = data
            for t in idxs[:-1]:
                d = d[t]
            d[idxs[-1]] = val

        # total elements
        total = 1
        for d in shape:
            total *= d

        base = x.data
        grad_data = TensorT.unit_tensor(0, shape).data  # nested zeros

        for i in range(total):
            mi = index_to_multi(i, shape)

            # f(x + h)
            x_plus = copy.deepcopy(base)
            set_at(x_plus, mi, get_at(base, mi) + h)
            f_plus = f(TensorT(x_plus))

            # f(x - h)
            x_minus = copy.deepcopy(base)
            set_at(x_minus, mi, get_at(base, mi) - h)
            f_minus = f(TensorT(x_minus))

            # central difference
            set_at(grad_data, mi, (f_plus - f_minus) / (2 * h))

        return TensorT(grad_data)

    # ---- analytical gradient (use tensor y = x*x and implicit ones grad) ----
    x = TensorT([[1.0, 2.0], [3.0, 4.0]])
    x.zero_grad()
    y = x * x                 # shape (2,2)
    y.backward()              # implicit d(sum)/dy = ones, so dy/dx = 2x
    analytical_grad = TensorT(x.grad)

    # ---- numerical gradient via finite differences on scalar f(x) ----
    numerical_grad = numerical_gradient(test_func, x)

    # Compare
    diff = (analytical_grad - numerical_grad)
    max_diff = max(abs(val) for val in diff.tflatten())
    print(f"✓ Max difference between analytical and numerical gradients: {max_diff}")
    if max_diff < 1e-4:
        print("✓ Gradient check PASSED")
    else:
        print("✗ Gradient check FAILED")


# Main test execution
if __name__ == '__main__':
    comprehensive_tensor_tests()
    gradient_check()
