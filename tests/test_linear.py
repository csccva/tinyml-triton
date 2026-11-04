import torch
import triton
import triton.language as tl
import pytest

from tinyml_triton.kernels.matmul import linear_kernel

@pytest.mark.parametrize("M, N, K", [
    (1, 1, 1),
    (16, 24, 8),
    (31, 33, 17),
    (64, 64, 64),
    (130, 127, 129),
    (256, 512, 128),
    (930, 727, 1129),
])
def test_linear(M, N, K):
    #M, N, K = 16, 32, 64
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = 32

    X = torch.randn((M,K), device="xpu", dtype=torch.float32)
    W = torch.randn((K,N), device="xpu", dtype=torch.float32)
    Y = torch.zeros((M,N), device="xpu", dtype=torch.float32)
    b = torch.randn((N), device="xpu", dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M,BLOCK_M),triton.cdiv(N,BLOCK_N),)

    linear_kernel[grid](
        X, W, b, Y,
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K
    )

    expected = torch.matmul(X,W)+b

    torch.xpu.synchronize()

    torch.testing.assert_close(Y, expected, rtol=1e-4, atol=1e-4)