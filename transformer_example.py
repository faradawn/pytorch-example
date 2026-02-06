"""
Example of defining a transformer block using Transformer Engine (TE)

This example demonstrates:
1. Using TE_TransformerLayer (fused LayerNorm + Attention + MLP)
2. Enabling FP8 autocast for optimized kernels
"""

import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe


def example_with_fp8():
    """Using TE_TransformerLayer with FP8 autocast for optimized kernels"""
    # Define model dimensions
    hidden_size = 768
    ffn_hidden_size = 3072
    num_attention_heads = 12
    
    # Create TransformerLayer
    transformer_layer = te.TransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
        layernorm_epsilon=1e-5,
        attention_dropout=0.1,
        hidden_dropout=0.1,
    ).cuda()
    
    # Create FP8 recipe
    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.E4M3,  # or recipe.Format.HYBRID
    )
    
    # Example input
    batch_size = 2
    seq_length = 512
    x = torch.randn(batch_size, seq_length, hidden_size).cuda()
    
    # IMPORTANT: Enable low precision explicitly
    # TE picks optimized kernels for enabled format
    with te.autocast(enabled=True, recipe=fp8_recipe):
        x = transformer_layer(x)  # Uses optimized FP8 kernels
    
    print(f"Output shape with FP8: {x.shape}")
    return x




if __name__ == "__main__":
    
    example_with_fp8()
    