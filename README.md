# PyTorch Optimization Examples

This repository demonstrates progressive optimization techniques for transformer models, from baseline to state-of-the-art implementations. It has been tested on Blackwell GPUs.

## Code Structure

### Optimization Progression

| Step | File | Optimization Type | Primary Speedup Source |
|------|------|-------------------|------------------------|
| **0. Baseline** | `0_baseline.py` | None | Standard CUDA kernels (FP32) |
| **1. CuBLAS** | `1_cublas.py` | Arithmetic | Tensor Cores (TF32/FP16) |
| **2. SDPA** | `2_sdpa.py` | Memory | Kernel Fusion (FlashAttention) |
| **3. Compile** | `3_compile.py` | Graph | Horizontal/Vertical Fusion (Triton) |
| **4. Transformer Engine** | `4_transformer_engine.py` | Precision | FP8 Transformer Engine |

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

## Usage

Run each optimization step individually:
```bash
python benchmark.pys
```

Example result on Blackwell GB10 Chip.
```
======================================================================
Model                                  ms/step    Speedup
======================================================================
Baseline (FP32)                           6.64   baseline
CuBLAS TF32                               3.31      2.00x
SDPA (FP32)                               3.72      1.79x
Compiled (FP32)                           2.96      2.24x
Transformer Engine (FP8)                  2.30      2.89x
======================================================================
```
