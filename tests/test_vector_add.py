import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    X,
    Y,
    Z,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X + offsets, mask=mask, other=0.0)
    y = tl.load(Y + offsets, mask=mask, other=0.0)

    tl.store(Z + offsets, x + y, mask=mask)


def main():
    n = 10_000_003
    device = "xpu"

    x = torch.randn(n, device=device)
    y = torch.randn(n, device=device)
    z = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    vector_add_kernel[grid](
        x, y, z, n,
        BLOCK_SIZE=1024,
    )

    torch.xpu.synchronize()

    reference = x.cpu() + y.cpu()
    result = z.cpu()
    max_error = (result - reference).abs().max().item()

    print(f"Device: {torch.xpu.get_device_name(0)}")
    print(f"Elements: {n:,}")
    print(f"Max absolute error: {max_error}")
    print(f"Test passed: {torch.allclose(result, reference, rtol=1e-5, atol=1e-6)}")

    assert torch.allclose(result, reference, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    main()
