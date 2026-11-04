import torch
import triton
import triton.language as tl
import pytest

from tinyml_triton.kernels.elementwise import softmax_kernel

@pytest.mark.parametrize("M,N", [(1,1), (16,17), (31,29), (64,65), (130,129), (256,257), (930,931)])
def test_softmax(M,N):
    #M, N, K = 16, 32, 64
    BLOCK_M=16
    BLOCK_N=triton.next_power_of_2(N) 

    S = torch.randn((M,N), device="xpu", dtype=torch.float32) #torch.randn((M,K), device="xpu", dtype=torch.float32)
    P = torch.zeros((M,N), device="xpu", dtype=torch.float32)

    grid = lambda meta: (M,)
    eps=1.0e-5

    softmax_kernel[grid](
        S, P,
        M, N,
        BLOCK_N
    )

    expected = torch.softmax(S, dim=1)

    torch.xpu.synchronize()

    torch.testing.assert_close(P, expected, rtol=1e-4, atol=1e-4)