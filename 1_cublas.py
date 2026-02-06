"""
1. CuBLAS: Tensor Core Acceleration

This optimization enables Tensor Cores by:
- Using TF32 precision for FP32 weights (NVIDIA Ampere+)
- Using Automatic Mixed Precision (AMP) with FP16
- GEMMs now use Tensor Cores via cuBLAS/cuBLASLt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualAttention(nn.Module):
    """Same manual attention as baseline, but will benefit from Tensor Cores"""
    
    def forward(self, q, k, v):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


class LayerNorm(nn.Module):
    """Simple LayerNorm implementation"""
    
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = 1e-5
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class MLP(nn.Module):
    """Simple MLP with GELU activation"""
    
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class CublasTransformerBlock(nn.Module):
    """Transformer block that benefits from Tensor Core acceleration"""
    
    def __init__(self, dim, num_heads, mlp_dim):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.ln2 = LayerNorm(dim)
        self.attention = ManualAttention()
        self.mlp = MLP(dim, mlp_dim)
        
        # Create Q, K, V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        # Attention block
        residual = x
        x = self.ln1(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        x = self.attention(q, k, v)
        x = x + residual
        
        # MLP block
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x + residual
        
        return x


def run_cublas_fp32():
    """Run with TF32 enabled (FP32 weights, Tensor Core math)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # Enable TF32 for NVIDIA Ampere+ (Speedup for FP32 weights)
    torch.set_float32_matmul_precision('high')
    
    # Model parameters
    batch_size = 8
    seq_len = 512
    dim = 768
    num_heads = 12
    mlp_dim = 3072
    steps = 10
    
    # Create model
    model = CublasTransformerBlock(dim, num_heads, mlp_dim).to(device)
    model.eval()
    
    # Create input
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
    
    # Warmup
    with torch.no_grad():
        _ = model(x)
    
    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        for _ in range(steps):
            x = model(x)
    end.record()
    torch.cuda.synchronize()
    
    elapsed = start.elapsed_time(end)
    print(f"CuBLAS TF32 (FP32 weights, Tensor Cores): {elapsed:.2f} ms for {steps} steps")
    print(f"Average: {elapsed/steps:.2f} ms per step")


def run_cublas_fp16():
    """Run with Automatic Mixed Precision (FP16)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # Enable TF32 for NVIDIA Ampere+
    torch.set_float32_matmul_precision('high')
    
    # Model parameters
    batch_size = 8
    seq_len = 512
    dim = 768
    num_heads = 12
    mlp_dim = 3072
    steps = 10
    
    # Create model
    model = CublasTransformerBlock(dim, num_heads, mlp_dim).to(device)
    model.eval()
    
    # Create input
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float16)
    
    # Warmup
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _ = model(x)
    
    # Benchmark with AMP
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        # Use Automatic Mixed Precision (AMP)
        # GEMMs now use Tensor Cores via cuBLAS/cuBLASLt
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for _ in range(steps):
                x = model(x)
    end.record()
    torch.cuda.synchronize()
    
    elapsed = start.elapsed_time(end)
    print(f"CuBLAS FP16 (AMP, Tensor Cores): {elapsed:.2f} ms for {steps} steps")
    print(f"Average: {elapsed/steps:.2f} ms per step")


if __name__ == "__main__":
    print("=" * 60)
    print("CuBLAS Optimization: Tensor Core Acceleration")
    print("=" * 60)
    print("\n1. TF32 Mode (FP32 weights, Tensor Core math):")
    run_cublas_fp32()
    print("\n2. FP16 Mode (AMP, Tensor Cores):")
    run_cublas_fp16()
