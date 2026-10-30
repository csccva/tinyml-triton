# Operation Inventory

## 1. Purpose

This document defines the tensor operations required to implement GPT-2 using custom Triton GPU kernels.

Operations are developed incrementally, independently validated against PyTorch reference implementations, and eventually assembled into Transformer decoder blocks.

The implementation priority is correctness before performance.

## 2. Operation Development Roadmap

### 2.1 Elementwise Operations

Fundamental tensor operations used throughout neural-network computations.

| Operation | Description | Status |
|---|---|---|
| Addition | Elementwise tensor addition | Planned |
| Subtraction | Elementwise tensor subtraction | Planned |
| Multiplication | Elementwise tensor multiplication | Planned |
| Scalar operations | Tensor operations involving scalar values | Planned |

### 2.2 Reduction Operations

Operations that reduce one or more tensor dimensions.

| Operation | Description | Status |
|---|---|---|
| Sum | Sum of tensor elements along a dimension | Planned |
| Maximum | Maximum value along a dimension | Planned |

### 2.3 Embedding Operations

Operations used to convert token IDs into vector representations.

| Operation | Description | Status |
|---|---|---|
| Embedding lookup | Retrieve embedding vectors using integer indices | Planned |

### 2.4 Matrix Operations

Operations used for linear transformations and projections.

| Operation | Description | Status |
|---|---|---|
| Matrix multiplication | General matrix multiplication (GEMM) | Planned |
| Linear projection | Matrix multiplication followed by bias addition | Planned |

### 2.5 Activation Operations

Nonlinear functions used in neural-network layers.

| Operation | Description | Status |
|---|---|---|
| GELU | GPT-2 GELU approximation (`gelu_new`) | Planned |

### 2.6 Normalization Operations

Operations used to normalize activations.

| Operation | Description | Status |
|---|---|---|
| Layer normalization | Normalize tensor values across the hidden dimension | Planned |

### 2.7 Softmax Operations

Operations used to convert scores into normalized probability distributions.

| Operation | Description | Status |
|---|---|---|
| Stable softmax | Numerically stable row-wise softmax | Planned |

### 2.8 Tensor Layout Operations

Operations required to organize tensors for Transformer computations.

Some layout transformations may be implemented using tensor views and stride metadata rather than custom GPU kernels.

| Operation | Description | Status |
|---|---|---|
| Reshape | Change tensor dimensions | Planned |
| Transpose | Change tensor dimension ordering | Planned |
| Indexing | Access tensor elements or slices | Planned |
| Head organization | Organize tensors into attention heads | Planned |

### 2.9 Attention Operations

Attention is developed incrementally, progressing from a simple implementation to the causal multi-head attention required by GPT-2.

| Operation | Description | Status |
|---|---|---|
| Single-head non-causal attention | Separate score computation, scaling, softmax, and weighted value aggregation | Planned |
| Single-head non-causal Flash Attention | Tiled attention with online softmax | Planned |
| Multi-head non-causal Flash Attention | Flash Attention across multiple attention heads | Planned |
| Causal multi-head Flash Attention | Multi-head Flash Attention with causal masking | Planned |

### 2.10 Output Operations

Operations required to produce the next token during autoregressive generation.

| Operation | Description | Status |
|---|---|---|
| Argmax | Select the index of the maximum logit | Planned |
| Sampling | Select a token using a probability distribution | Planned |

## 3. GPT-2 Model Configuration

The initial target is GPT-2 Small.

| Parameter | Value |
|---|---:|
| Vocabulary size | 50,257 |
| Context length | 1,024 |
| Number of decoder blocks | 12 |
| Hidden dimension | 768 |
| Number of attention heads | 12 |
| Attention head dimension | 64 |
| MLP intermediate dimension | 3,072 |

The model uses learned token and positional embeddings, pre-layer normalization, causal self-attention, and an MLP within each Transformer decoder block.

The token embedding weights are tied to the output projection weights.

## 4. Implementation Strategy

Operations are implemented in the order defined in Section 2.

Each operation must be independently validated before being used in a more complex component.

After the required operations are implemented, they will be assembled incrementally:

1. Implement and validate a single Transformer decoder block.
2. Validate the decoder block against a PyTorch reference.
3. Assemble multiple decoder blocks.
4. Build the complete 12-block GPT-2 Small model.
5. Implement the output projection and logits computation.
6. Implement greedy autoregressive generation.
7. Introduce KV caching as a later optimization.

## 5. Initial Implementation Constraints

- Initial implementation uses FP32.
- Inference only; training and backpropagation are outside the initial scope.
- Triton kernels execute on the Intel GPU through the XPU backend.
- PyTorch is permitted for tensor management, model orchestration, weight loading, and independent correctness references.
- PyTorch must not be used as a fallback for neural-network computations.
- Greedy decoding is implemented before more advanced sampling strategies.
- KV caching is postponed until the initial implementation is correct.
- Correctness is established before performance optimization.