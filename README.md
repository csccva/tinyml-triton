# tinyml-triton

An educational implementation of neural-network operations and GPT-2
inference using PyTorch and custom Triton kernels.

The project targets Intel GPUs through the Triton XPU backend. Its
primary goal is to explore GPU programming, understand the operations
behind Transformer models, and progressively implement GPT-2 inference
using custom GPU kernels.

## Current status

### Environment and GPU setup

-   [x] PyTorch XPU detects Intel Graphics (device ID 0x7d55) under
    WSL2.
-   [x] Custom Triton kernels execute on the Intel GPU.
-   [x] PyTorch reference implementations are used for numerical
    validation.

### Implemented and validated operations

-   [x] Elementwise operations: addition, subtraction, multiplication
    and division.
-   [x] Scalar operations: addition and multiplication.
-   [x] Sum reduction.
-   [x] Maximum reduction.
-   [x] Matrix multiplication (GEMM).
-   [x] Linear projection with bias.
-   [x] GELU-new activation.
-   [x] Layer normalization.
-   [x] Stable row-wise softmax.

### GPT-2 implementation

-   [ ] Tensor indexing, reshape and transpose.
-   [ ] Single-head attention.
-   [ ] Flash Attention using tiling and online softmax.
-   [ ] Multi-head causal attention.
-   [ ] Argmax.
-   [ ] Token and positional embedding lookup.
-   [ ] Assemble the GPT-2 decoder block.
-   [ ] Implement the complete GPT-2 Small forward pass.
-   [ ] Implement autoregressive text generation.
-   [ ] Validate numerical agreement with a reference implementation.

## GPT-2 target configuration

The initial implementation targets GPT-2 Small:

  Parameter                       Value
  ---------------------------- --------
  Vocabulary size                50,257
  Context length                  1,024
  Hidden dimension                  768
  Number of decoder blocks           12
  Attention heads                    12
  Head dimension                     64
  MLP intermediate dimension      3,072

Initial implementation constraints:

-   Inference only; no training or backpropagation.
-   FP32 precision.
-   Greedy decoding.
-   PyTorch reference implementation for correctness validation.
-   Custom Triton kernels for GPU operations.
-   KV cache and performance optimizations will be introduced in later
    stages.

## Development environment

The project has been tested with:

-   Python 3.12.3
-   PyTorch 2.7.1+XPU
-   Triton 3.3.1
-   Pytest 9.2.0
-   Intel Graphics (device ID 0x7d55)
-   Ubuntu 24.04 under WSL2

These are the verified test versions, not a guarantee that other
configurations behave identically.

## Testing

Run the complete test suite:

``` bash
python -m pytest -v
```

Latest validation:

-   77 tests passed.
-   No failing tests.
-   Execution time: 27.09 seconds.

Tests compare custom Triton kernel outputs against independent PyTorch
reference implementations.

## Project structure

``` text
tinyml-triton/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── OPERATIONS.md
│   └── VALIDATION.md
├── benchmarks/
│   └── benchmark_matmul.py
├── src/
│   └── tinyml_triton/
│       └── kernels/
│           ├── elementwise.py
│           ├── reduction.py
│           └── matmul.py
├── tests/
├── pyproject.toml
├── README.md
└── LICENSE
```

## Development philosophy

Operations are implemented incrementally, following this workflow:

1.  Understand the mathematical operation.
2.  Implement the Triton kernel.
3.  Validate against an independent PyTorch reference.
4.  Test different tensor dimensions and edge cases.
5.  Benchmark when relevant.
6.  Document the implementation and validation.
7.  Commit the completed milestone.

Correctness takes priority over performance optimization. Individual
operations are validated before being assembled into larger Transformer
components.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
