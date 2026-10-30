import triton
import triton.language as tl


@triton.jit
def max_reduction_small_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Generate offsets.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE +tl.arange(0,BLOCK_SIZE)
    # 2. Create a mask for valid elements.
    mask= offsets < n_elements
    # 3. Load input elements.
    x = tl.load(x_ptr+offsets,mask=mask, other=float("-inf"))
    # 4. Calculate the sum using tl.sum().
    max = tl.max(x)
    
    # 5. Store the result.
    tl.store(output_ptr,max)

@triton.jit
def sum_reduction_small_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Generate offsets.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE +tl.arange(0,BLOCK_SIZE)
    # 2. Create a mask for valid elements.
    mask= offsets < n_elements
    # 3. Load input elements.
    x = tl.load(x_ptr+offsets,mask=mask, other=0)
    # 4. Calculate the sum using tl.sum().
    sum = tl.sum(x)

    # 5. Store the result.
    tl.store(output_ptr,sum)