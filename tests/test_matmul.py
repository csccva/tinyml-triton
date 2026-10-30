import torch
import triton
import triton.language as tl
import pytest

from tinyml_triton.kernels.matmul import matmul_kernel

@pytest.mark.parametrize("M, N, K", [
    (1, 1, 1),
    (16, 24, 8),
    (31, 33, 17),
    (64, 64, 64),
    (130, 127, 129),
    (256, 512, 128),
    (930, 727, 1129),
])
def test_matmul(M, N, K):
    # M, N, K = 930, 727, 1129

    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = 32

    A = torch.randn((M,K), device="xpu", dtype=torch.float32)
    B = torch.randn((K,N), device="xpu", dtype=torch.float32)
    C = torch.zeros((M,N), device="xpu", dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M,BLOCK_M),triton.cdiv(N,BLOCK_N),)

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K
    )

    expected = torch.matmul(A,B)

    torch.xpu.synchronize()

    torch.testing.assert_close(C, expected, rtol=1e-4, atol=1e-4)