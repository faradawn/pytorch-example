"""
0. Baseline: The "Pure" PyTorch Model

This demonstrates the naive manual attention implementation.
It creates many intermediate tensors in global memory, which is slow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualAttention(nn.Module):
    """The 'Naive' way: Lots of memory bandwidth usage"""
    
    def forward(self, q, k, v):
        d_k = q.size(-1)
        # Step 1: Compute attention scores (creates intermediate tensor)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        # Step 2: Apply softmax (creates another intermediate tensor)
        attn = F.softmax(scores, dim=-1)
        # Step 3: Apply attention to values (creates final tensor)
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


class BaselineTransformerBlock(nn.Module):
    """Baseline transformer block with manual attention"""
    
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


def run_baseline():
    """Run baseline model (FP32, no optimizations)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # Model parameters
    batch_size = 8
    seq_len = 512
    dim = 768
    num_heads = 12
    mlp_dim = 3072
    steps = 10
    
    # Create model
    model = BaselineTransformerBlock(dim, num_heads, mlp_dim).to(device)
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
    print(f"Baseline (FP32, Manual Attention): {elapsed:.2f} ms for {steps} steps")
    print(f"Average: {elapsed/steps:.2f} ms per step")


if __name__ == "__main__":
    run_baseline()
