# Validation Strategy

## 1. Purpose

This document defines the validation strategy for the custom Triton operations and GPT-2 model implementation.

Every operation must be validated against an independent PyTorch reference before it is used in more complex components.

Correctness, GPU execution, and performance are evaluated separately.

## 2. Validation Levels

### 2.1 Kernel Validation

Each Triton kernel is tested independently.

Validation includes:

- Comparing numerical results against an independent PyTorch reference.
- Testing supported tensor shapes and dimensions.
- Testing supported data types.
- Testing boundary conditions and relevant edge cases.
- Verifying correct behavior on the Intel GPU through the Triton XPU backend.

### 2.2 Operation Validation

The Python interface in `ops/` is tested independently of the underlying kernel implementation.

Validation includes:

- Input validation and argument handling.
- Correct kernel selection and launch configuration.
- Correct output shape and data type.
- Numerical agreement with the PyTorch reference.
- Correct handling of supported edge cases.

### 2.3 Integration Validation

Multiple operations are combined and tested together.

Validation includes:

- Correct interaction between operations.
- Correct tensor shapes and data flow.
- Numerical comparison against equivalent PyTorch implementations.
- Validation of intermediate results.

### 2.4 Model Validation

Complete model components are validated incrementally.

The validation sequence is:

1. Single Transformer decoder block.
2. Multiple decoder blocks.
3. Complete GPT-2 Small model.
4. Output logits.
5. Autoregressive generation.

Intermediate tensors and logits must be compared against the PyTorch reference, not just the final generated tokens.

## 3. Numerical Correctness

Numerical comparisons use explicit absolute and relative tolerances.

The comparison is based on:

\[
|x_{\text{Triton}}-x_{\text{reference}}|
\leq
\text{atol}+\text{rtol}\cdot|x_{\text{reference}}|
\]

Tolerance values must be selected according to the operation, data type, and numerical behavior.

Validation should include:

- Maximum absolute error.
- Maximum relative error.
- Mean absolute error.
- Detection of NaN and Inf values.

Tolerance values must be documented for each operation.

## 4. GPU Execution Validation

Tests must verify that Triton kernels execute on the intended Intel GPU through the XPU backend.

The test environment must not silently fall back to CPU execution or PyTorch computation.

Tests requiring the Intel GPU must explicitly detect whether the required device and backend are available.

GPU execution is asynchronous. Tests must ensure that GPU computation has completed before reading results or comparing tensors.

## 5. Performance Validation

Performance measurements are maintained separately from correctness tests.

Benchmarks are performed at two levels:

- **Kernel benchmarks:** Measure individual Triton kernel execution times.
- **Model benchmarks:** Measure model component execution and end-to-end inference performance.

Benchmarking must account for asynchronous GPU execution, including appropriate synchronization or device-event timing.

Performance comparisons should use consistent hardware, input shapes, data types, and execution conditions.

## 6. Validation Principles

1. Every operation must pass independent correctness tests before integration.
2. Kernel tests and operation interface tests must be kept separate.
3. Integration tests must validate intermediate results, not only final outputs.
4. GPU execution must be verified explicitly.
5. No silent CPU or PyTorch fallback is permitted.
6. Performance optimization must not compromise numerical correctness.
7. Validation results and tolerances must be documented.