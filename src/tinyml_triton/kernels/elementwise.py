import triton
import triton.language as tl


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