"""
3. PyTorch Compile: Graph-Level Fusion

While SDPA fuses the attention head, torch.compile uses the OpenAI Triton
or C++ Inductor backend to fuse the other parts (LayerNorm, GELU, and
Residual connections).

This generates custom Triton kernels for the MLP and LN layers.
It "glues" the whole block together to avoid kernel launch overhead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedAttention(nn.Module):
    """Uses PyTorch's scaled_dot_product_attention"""
    
    def forward(self, q, k, v, is_causal=False):
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


class CompiledTransformerBlock(nn.Module):
    """Transformer block optimized for torch.compile"""
    
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
