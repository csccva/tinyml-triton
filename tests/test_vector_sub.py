# python -m pytest -v -s tests/test_vector_sub.py
import pytest
import torch
import triton

from tinyml_triton.kernels.elementwise import vector_sub_kernel


@pytest.mark.parametrize("n_elements", [
    1,
    256,
    257,
    4096,
    10_000_003,
])
def test_vector_sub(n_elements):
    # 1. Verify that the XPU is available.
    
    print("PyTorch:", torch.__version__)
    print("Triton:", triton.__version__)
    print("GPU available:", torch.xpu.is_available())

    if not torch.xpu.is_available():
        raise RuntimeError("PyTorch cannot access the Intel GPU")

    print("GPU:", torch.xpu.get_device_name(0))
    # 2. Create input tensors on the XPU.
    a = torch.randn(n_elements, device="xpu", dtype=torch.float32)
    b = torch.randn(n_elements, device="xpu", dtype=torch.float32)
    # 3. Allocate the output tensor.
    out = torch.empty_like(a)

    # 4. Calculate the Triton launch grid.
    # Hint: use triton.cdiv() and BLOCK_SIZE.
    BLOCK_SIZE = 256
    grid= lambda meta:(triton.cdiv(n_elements, BLOCK_SIZE),)

    # 5. Launch vector_add_kernel.
    vector_sub_kernel[grid](a,b,out,n_elements,BLOCK_SIZE=256)

    # 6. Synchronize the XPU.
    torch.xpu.synchronize()

    # 7. Calculate an independent PyTorch reference.
    expected = torch.sub(a,b)

    # 8. Compare the kernel output against the reference.
    max_error = (out - expected).abs().max().item()

    print("Maximum error:", max_error)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
