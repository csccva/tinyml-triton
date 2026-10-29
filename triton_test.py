import torch, triton, triton.language as tl

@triton.jit
def add(x, y, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

n = 4096
x = torch.randn(n, device="xpu"); y = torch.randn(n, device="xpu"); o = torch.empty_like(x)
add[(triton.cdiv(n, 256),)](x, y, o, n, BLOCK=256)
print(torch.allclose(o, x + y))