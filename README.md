# tinyml-triton

An educational implementation of neural-network operations and GPT-2
inference using PyTorch and custom Triton kernels.

The project targets Intel GPUs through the Triton XPU backend.

## Current status

- [x] PyTorch XPU detects Intel Graphics (device ID 0x7d55) under WSL2.
- [x] A custom Triton vector-add kernel executes on the GPU.
- [x] The vector-add output matches the PyTorch reference exactly.
- [ ] Implement and validate individual neural-network operations.
- [ ] Implement the complete GPT-2 forward pass.
- [ ] Validate numerical agreement with a reference implementation.

## Development environment

The initial GPU test was verified with:

- Python 3.12
- PyTorch 2.7.1+XPU
- Triton 3.3.1
- Intel Core Ultra 7 155H integrated GPU
- Ubuntu 24.04 under WSL2

These are the verified test versions, not a guarantee that other
configurations behave identically.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
