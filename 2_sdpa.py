"""
2. PyTorch SDPA: Fused Kernels

This moves from "Faster Math" to "Fewer Memory Trips."
SDPA replaces the manual Softmax/MatMul/Dropout with a single fused kernel
(like FlashAttention-2).

DRAM access reduced from 5x to 1x.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedAttention(nn.Module):
    """
    Uses PyTorch's scaled_dot_product_attention.
    Fuses: MatMul -> Scaling -> Masking -> Softmax -> Dropout -> MatMul
    """
    
    def forward(self, q, k, v, is_causal=False):
        # Fuses: MatMul -> Scaling -> Masking -> Softmax -> Dropout -> MatMul
        # DRAM access reduced from 5x to 1x
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


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


class SDPATransformerBlock(nn.Module):
    """Transformer block using SDPA for fused attention"""
    
    def __init__(self, dim, num_heads, mlp_dim, is_causal=False):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.ln2 = LayerNorm(dim)
        self.attention = OptimizedAttention()
        self.mlp = MLP(dim, mlp_dim)
        self.is_causal = is_causal
        
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
        x = self.attention(q, k, v, is_causal=self.is_causal)
        x = x + residual
        
        # MLP block
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x + residual
        
        return x


def run_sdpa():
    """Run with SDPA (fused attention kernels)"""
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
    model = SDPATransformerBlock(dim, num_heads, mlp_dim, is_causal=True).to(device)
    model.eval()
    
    # Create input
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float16)
    
    # Warmup
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            _ = model(x)
    
    # Benchmark with SDPA + AMP
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for _ in range(steps):
                x = model(x)
    end.record()
    torch.cuda.synchronize()
    
    elapsed = start.elapsed_time(end)
    print(f"SDPA (Fused Kernels + FP16): {elapsed:.2f} ms for {steps} steps")
    print(f"Average: {elapsed/steps:.2f} ms per step")
    print("\nNote: SDPA fuses attention operations, reducing DRAM access from 5x to 1x")


if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch SDPA: Fused Kernels")
    print("=" * 60)
    run_sdpa()
