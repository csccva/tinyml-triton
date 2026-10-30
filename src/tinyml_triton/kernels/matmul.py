
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 1. Identify the program's position in the output matrix.
    pid_m = tl.program_id(0)

    pid_n = tl.program_id(1)

    # 2. Calculate the row and column offsets.
    offs_m = pid_m*BLOCK_M + tl.arange(0,BLOCK_M)
    offs_n = pid_n*BLOCK_N + tl.arange(0,BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    c_mask = mask_m[:, None] & mask_n[None, :]
    
    

    # 4. Create masks for valid row and column indices.
    #    K will also need masking inside the loop.

    # 5. Initialize the accumulator.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 6. Iterate over the K dimension in blocks.
    for k in range(tl.cdiv(K,BLOCK_K)):
        # 3. Generate offsets along the reduction dimension.
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        # 7. Calculate the offsets for loading A and B.
        a_offsets = offs_m[:,None]*K+offs_k[None,:]
        a_mask = mask_m[:, None] & mask_k[None, :]
        b_offsets = offs_k[:,None]*N+offs_n[None,:]
        b_mask = mask_k[:, None] & mask_n[None, :]

        # 8. Load the corresponding tiles of A and B.
        a=tl.load(A_ptr+a_offsets,mask=a_mask,other=0.0)
        b=tl.load(B_ptr+b_offsets,mask=b_mask,other=0.0)
        # 9. Perform the matrix multiplication using tl.dot().
        accumulator=tl.dot(a,b,accumulator)
    
    # 11. Calculate the output offsets.
    offs_c = offs_m[:,None]*N + offs_n[None,:]
    c_mask = mask_m[:, None] & mask_n[None, :]
    # 12. Store the resulting tile of C
    tl.store(C_ptr+offs_c, accumulator,c_mask)