import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    tl.store(out_ptr + offsets, a + b, mask=mask)


def main():
    print("PyTorch:", torch.__version__)
    print("Triton:", triton.__version__)
    print("GPU available:", torch.xpu.is_available())

    if not torch.xpu.is_available():
        raise RuntimeError("PyTorch cannot access the Intel GPU")

    print("GPU:", torch.xpu.get_device_name(0))

    n = 4099
    a = torch.arange(n, device="xpu", dtype=torch.float32)
    b = torch.full((n,), 2.0, device="xpu", dtype=torch.float32)
    out = torch.empty_like(a)

    add_kernel[(triton.cdiv(n, 256),)](a, b, out, n, 256)

    torch.xpu.synchronize()

    expected = a + b
    max_error = (out - expected).abs().max().item()

    print("Maximum error:", max_error)
    print("Test passed:", torch.equal(out, expected))


if __name__ == "__main__":
    main()
