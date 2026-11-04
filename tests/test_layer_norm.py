import torch
import triton
import triton.language as tl
import pytest

from tinyml_triton.kernels.elementwise import layer_norm_kernel

@pytest.mark.parametrize("M,K", [(1,1), (16,17), (31,29), (64,65), (130,129), (256,257), (930,931)])
def test_layer_norm(M,K):
    #M, N, K = 16, 32, 64
    BLOCK_K = triton.next_power_of_2(K)

    X = - torch.rand((M,K), device="xpu", dtype=torch.float32) #torch.randn((M,K), device="xpu", dtype=torch.float32)
    Y = torch.zeros((M,K), device="xpu", dtype=torch.float32)
    gamma = torch.randn((K), device="xpu", dtype=torch.float32)
    beta = torch.randn((K), device="xpu", dtype=torch.float32)

    grid = lambda meta: (M,)
    eps=1.0e-5

    layer_norm_kernel[grid](
        X,gamma,beta,Y,
        M, K, eps,
        BLOCK_K
    )

    expected = torch.nn.functional.layer_norm( X, (K,), gamma, beta, eps)

    torch.xpu.synchronize()

    torch.testing.assert_close(Y, expected, rtol=1e-4, atol=1e-4)