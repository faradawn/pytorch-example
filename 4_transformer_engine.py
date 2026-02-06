"""
4. Transformer Engine: FP8 & Deep Integration

The final evolution. This uses Transformer Engine (TE) to handle FP8 scaling
factors dynamically, which is significantly faster than FP16 on H100/B200 GPUs.

TE replaces standard Layers with FP8-optimized versions and uses:
- CuBLASLt FP8 kernels
- TE's fused LayerNorm/GELU
"""

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("Warning: transformer_engine not available. Install with:")
    print("  pip install transformer-engine[pytorch]")


import torch
import torch.nn as nn
import torch.nn.functional as F


if TE_AVAILABLE:
    class TransformerEngineBlock(nn.Module):
        """
        Transformer block using Transformer Engine.
        TE replaces standard Layers with FP8-optimized versions.
        """
        
        def __init__(self, dim, num_heads, mlp_dim):
            super().__init__()
            # Use TE's TransformerLayer which includes:
            # - FP8-optimized attention
            # - FP8-optimized MLP
            # - Fused LayerNorm and GELU
            self.layer = te.TransformerLayer(
                hidden_size=dim,
                ffn_hidden_size=mlp_dim,
                num_attention_heads=num_heads,
                layernorm_epsilon=1e-5,
                attention_dropout=0.0,
                hidden_dropout=0.0,
            )
        
        def forward(self, x):
            return self.layer(x, attention_mask=None)
else:
    # Placeholder class when TE is not available
    TransformerEngineBlock = None
