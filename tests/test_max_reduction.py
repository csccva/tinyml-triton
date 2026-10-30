import pytest
import torch
import triton
import math

from tinyml_triton.kernels.reduction import max_reduction_small_kernel

@pytest.mark.parametrize("shape", [1, 128, 256, 512, 1024])

def test_max_reduction(shape):
    print("PyTorch:", torch.__version__)
    print("Triton:", triton.__version__)
    print("GPU available:", torch.xpu.is_available())

    if not torch.xpu.is_available():
        raise RuntimeError("PyTorch cannot access the Intel GPU")

    print("GPU:", torch.xpu.get_device_name(0))

    # 1. Calculate the total number of elements.
    n_elements = shape

    # 2. Create two random FP32 tensors on XPU with the given shape.
    a = torch.randn(shape, device="xpu", dtype=torch.float32)

    # 3. Allocate the output tensor.
    out = torch.empty((1,), device="xpu", dtype=torch.float32)
    # 4. Define BLOCK_SIZE and calculate the launch grid.
    # Hint: use the total number of elements, not the dimensions.
    BLOCK_SIZE = 1024
    grid= lambda meta:(triton.cdiv(n_elements, BLOCK_SIZE),)

    # 5. Launch the existing vector_add_kernel.
    max_reduction_small_kernel[grid](a,out,n_elements,BLOCK_SIZE)

    # 6. Synchronize XPU.
    torch.xpu.synchronize()

    # 7. Calculate the PyTorch reference.
    expected=torch.max(a)

    # 8. Compare results.
    max_error = (out[0] - expected).abs().max().item()

    print("Maximum error:", max_error)
    torch.testing.assert_close(out[0], expected, rtol=1e-5, atol=1e-6)