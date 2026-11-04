import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    S_ptr,
    P_ptr,
    M, N,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)

    # Generate column indices
    offs_n=tl.arange(0,BLOCK_N)

    # Create mask
    mask_n = offs_n < N

    # Calculate row offset

    offs = pid*N + offs_n

    # Load scores

    scores=tl.load(S_ptr+offs,mask=mask_n,other=float("-inf"))

    # Calculate row maximum

    row_max= tl.max(scores, axis=0)

    # Subtract maximum and calculate exponentials
    diff=scores-row_max

    # Calculate exponential sum
    sum_exp=tl.sum(tl.exp(diff), axis=0)

    #  Normalize
    probabilities=tl.exp(diff)/sum_exp

    #  Store probabilities
    tl.store(P_ptr+offs,probabilities,mask=mask_n)


@triton.jit
def layer_norm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    M, K, eps,
    BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_K)
    mask = offs_c < K
    offs = pid * K + offs_c
    x=tl.load(x_ptr+offs,mask=mask,other=0.0)
    gamma = tl.load(gamma_ptr + offs_c, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offs_c, mask=mask, other=0.0)
    mean=tl.sum(x)/K
    diff = x-mean
    vari = tl.sum(tl.where(mask, diff * diff, 0.0)) / K
    x=(x-mean)*tl.rsqrt(vari +eps)
    x=x*gamma+beta

    tl.store(out_ptr + offs, x, mask=mask)

@triton.jit
def gelu_new_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n*BLOCK_N + tl.arange(0,BLOCK_N)
    mask_n=offs_n < N
    x=tl.load(x_ptr+offs_n,mask=mask_n,other=0.0)

    z = tl.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * x * x * x)
    tanhz = (2.0 / (1.0 + tl.exp(-2.0 * z))) - 1.0
    gelu = 0.5 * x * (1.0 + tanhz)
    tl.store(out_ptr + offs_n, gelu, mask=mask_n)


@triton.jit
def vector_scalar_mul_kernel(
    x_ptr,
    a,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Calculate the program ID.
    pid = tl.program_id(0)

    # 2. Generate the offsets for this program.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. Create a mask for valid elements.
    mask = offsets < n_elements

    # 4. Load x and y using the mask.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # 5. Add the loaded values.
    z=x*a
    # 6. Store the result using the mask.
    tl.store(output_ptr+ offsets, z, mask=mask)

@triton.jit
def vector_scalar_add_kernel(
    x_ptr,
    a,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Calculate the program ID.
    pid = tl.program_id(0)

    # 2. Generate the offsets for this program.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. Create a mask for valid elements.
    mask = offsets < n_elements

    # 4. Load x and y using the mask.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # 5. Add the loaded values.
    z=x+a
    # 6. Store the result using the mask.
    tl.store(output_ptr+ offsets, z, mask=mask)

@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Calculate the program ID.
    pid = tl.program_id(0)

    # 2. Generate the offsets for this program.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. Create a mask for valid elements.
    mask = offsets < n_elements

    # 4. Load x and y using the mask.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # 5. Add the loaded values.
    z=x+y
    # 6. Store the result using the mask.
    tl.store(output_ptr+ offsets, z, mask=mask)

@triton.jit
def vector_sub_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Calculate the program ID.
    pid = tl.program_id(0)

    # 2. Generate the offsets for this program.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. Create a mask for valid elements.
    mask = offsets < n_elements

    # 4. Load x and y using the mask.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # 5. Add the loaded values.
    z=x-y
    # 6. Store the result using the mask.
    tl.store(output_ptr+ offsets, z, mask=mask)

@triton.jit
def vector_mul_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Calculate the program ID.
    pid = tl.program_id(0)

    # 2. Generate the offsets for this program.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. Create a mask for valid elements.
    mask = offsets < n_elements

    # 4. Load x and y using the mask.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # 5. Add the loaded values.
    z=x*y
    # 6. Store the result using the mask.
    tl.store(output_ptr+ offsets, z, mask=mask)

@triton.jit
def vector_div_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Calculate the program ID.
    pid = tl.program_id(0)

    # 2. Generate the offsets for this program.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. Create a mask for valid elements.
    mask = offsets < n_elements

    # 4. Load x and y using the mask.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # 5. Add the loaded values.
    z=x/y
    # 6. Store the result using the mask.
    tl.store(output_ptr+ offsets, z, mask=mask)