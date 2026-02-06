# PyTorch Optimization Examples

This repository demonstrates progressive optimization techniques for transformer models, from baseline to state-of-the-art implementations.

## Code Structure

### Optimization Progression

| Step | File | Optimization Type | Primary Speedup Source |
|------|------|-------------------|------------------------|
| **0. Baseline** | `0_baseline.py` | None | Standard CUDA kernels (FP32) |
| **1. CuBLAS** | `1_cublas.py` | Arithmetic | Tensor Cores (TF32/FP16) |
| **2. SDPA** | `2_sdpa.py` | Memory | Kernel Fusion (FlashAttention) |
| **3. Compile** | `3_compile.py` | Graph | Horizontal/Vertical Fusion (Triton) |
| **4. Transformer Engine** | `4_transformer_engine.py` | Precision | 8-bit Compute (H100/Hopper) |

### File Descriptions

- **`0_baseline.py`**: Pure PyTorch model with manual attention implementation. Demonstrates the naive approach that creates many intermediate tensors in global memory.

- **`1_cublas.py`**: Enables Tensor Core acceleration through:
  - TF32 precision for FP32 weights (NVIDIA Ampere+)
  - Automatic Mixed Precision (AMP) with FP16
  - GEMM operations use Tensor Cores via cuBLAS/cuBLASLt

- **`2_sdpa.py`**: Uses PyTorch's `scaled_dot_product_attention` to fuse attention operations:
  - Replaces manual Softmax/MatMul/Dropout with a single fused kernel
  - Reduces DRAM access from 5x to 1x (FlashAttention-style)

- **`3_compile.py`**: Graph-level optimization using `torch.compile`:
  - Generates custom Triton kernels for MLP and LayerNorm layers
  - Fuses the entire block to avoid kernel launch overhead
  - Uses OpenAI Triton or C++ Inductor backend

- **`4_transformer_engine.py`**: Final optimization using Transformer Engine:
  - FP8 dynamic scaling for H100/B200 GPUs
  - CuBLASLt FP8 kernels + fused LayerNorm/GELU
  - Deep integration with NVIDIA hardware

## Setup

### Docker Container
```bash
docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/pytorch:25.11-py3
```

**Docker flags:**
- `--gpus all`: Enables GPU access (required for NVIDIA containers)
- `-it`: Interactive terminal mode
- `--rm`: Automatically removes container when it exits
- `-v $(pwd):/workspace`: Mounts current directory to `/workspace` in container
- `-w /workspace`: Sets working directory to `/workspace`

### Dependencies
- PyTorch 2.0+ (included in NVIDIA container)
- Transformer Engine (optional, for step 4): `pip install transformer-engine[pytorch]`

## Usage

Run each optimization step individually:
```bash
python 0_baseline.py
python 1_cublas.py
python 2_sdpa.py
python 3_compile.py
python 4_transformer_engine.py
```

Each script includes benchmarking code to measure performance improvements.