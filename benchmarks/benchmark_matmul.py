import torch
import triton
import statistics
from tinyml_triton.kernels.matmul import matmul_kernel


configs = [
    triton.Config(
        {'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64},
        num_warps=8,
        num_stages=2,
    ),
]

autotuned_matmul = triton.autotune(configs=configs,key=['M', 'N', 'K'],)(matmul_kernel)


# Preserve original method.
_original_bench = autotuned_matmul._bench


results = {}

def logged_bench(*args, **kwargs):
    timings = _original_bench(*args, **kwargs)

    median_time = statistics.median(timings)
    min_time = min(timings)
    max_time = max(timings)

    # Extract the GEMM dimensions from the kernel arguments.
    M, N, K = args[3:6]
    shapes = (M, N, K)

    # Extract the configuration being benchmarked.
    config = kwargs.get("config")

    # results.setdefault(config, []).append({
    #     "median_ms": median_time,
    #     "min_ms": min_time,
    #     "max_ms": max_time,
    # })

    results.setdefault(shapes, []).append({
        "config": config,
        "median_ms": median_time,
        "min_ms": min_time,
        "max_ms": max_time,
    })

    print(f"Shapes:    {key}")
    print(f"Config: {config}")
    print(f"Timings: {timings}")
    print(f"Median: {median_time:.4f} ms")
    print(f"Min:    {min_time:.4f} ms")
    print(f"Max:    {max_time:.4f} ms")

    return timings

print(autotuned_matmul._bench)
# Apply monkey patch.
autotuned_matmul._bench = logged_bench
print(autotuned_matmul._bench)

def test_matmul(M, N, K):
    A = torch.randn((M,K), device="xpu", dtype=torch.float32)
    B = torch.randn((K,N), device="xpu", dtype=torch.float32)
    C = torch.zeros((M,N), device="xpu", dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),triton.cdiv(N, meta["BLOCK_N"]),)

    autotuned_matmul[grid](A, B, C,M, N, K)
    
    expected = torch.matmul(A,B)

    torch.xpu.synchronize()

    torch.testing.assert_close(C, expected, rtol=1e-4, atol=1e-4)
    print(results)

if __name__ == "__main__":
    test_matmul(256, 256, 128)
    print("Selected configuration:", autotuned_matmul.best_config)