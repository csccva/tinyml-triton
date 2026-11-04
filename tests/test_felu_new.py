import torch
import triton
import triton.language as tl
import pytest

from tinyml_triton.kernels.elementwise import gelu_new_kernel

@pytest.mark.parametrize("N", [1, 16, 31, 64, 130, 256, 930])
def test_gelu_new(N):
    #M, N, K = 16, 32, 64
    BLOCK_N = 32

    X = torch.randn((N), device="xpu", dtype=torch.float32)
    Y = torch.zeros((N), device="xpu", dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(N,BLOCK_N),)

    gelu_new_kernel[grid](
        X, Y,
        N,
        BLOCK_N
    )

    expected = 0.5 * X * (1.0 + torch.tanh((2.0 / torch.pi) ** 0.5 * (X + 0.044715 * X**3))) #torch.gelu_new(X)

    torch.xpu.synchronize()

    torch.testing.assert_close(Y, expected, rtol=1e-4, atol=1e-4)