"""
Unified Benchmarking Script

This script benchmarks all transformer block implementations with the same
parameters and workload to compare performance across optimization levels.
"""

import torch
import importlib.util
import sys
from typing import Dict, Optional

# Check for CUDA availability - this script requires CUDA
if not torch.cuda.is_available():
    print("=" * 80)
    print("ERROR: CUDA is not available!")
    print("=" * 80)
    print("This benchmark script requires CUDA to run.")
    print("Please run this script on a machine with an NVIDIA GPU and CUDA installed.")
    print("\nTo check CUDA availability, run: python -c 'import torch; print(torch.cuda.is_available())'")
    sys.exit(1)

# Note: Files start with numbers, so we use importlib to load them
def import_from_file(file_path, class_name):
    spec = importlib.util.spec_from_file_location("module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

# Import transformer blocks
BaselineTransformerBlock = import_from_file('0_baseline.py', 'BaselineTransformerBlock')
CublasTransformerBlock = import_from_file('1_cublas.py', 'CublasTransformerBlock')
SDPATransformerBlock = import_from_file('2_sdpa.py', 'SDPATransformerBlock')
CompiledTransformerBlock = import_from_file('3_compile.py', 'CompiledTransformerBlock')

# Transformer Engine (may fail if not installed)
try:
    TransformerEngineBlock = import_from_file('4_transformer_engine.py', 'TransformerEngineBlock')
except Exception as e:
    TransformerEngineBlock = None
    print(f"Warning: Could not import TransformerEngineBlock: {e}")


def benchmark_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    steps: int,
    name: str,
    autocast_context: Optional[object] = None,
    warmup_steps: int = 3,
) -> float:
    """
    Benchmark a model with the given input and number of steps.
    
    Args:
        model: The model to benchmark
        x: Input tensor
        steps: Number of benchmark steps
        name: Name of the model for printing
        autocast_context: Optional autocast context manager or callable that returns one
        warmup_steps: Number of warmup steps
    
    Returns:
        Average time per step in milliseconds
    """
    model.eval()
    device = x.device
    
    # Warmup
    with torch.no_grad():
        if autocast_context is not None:
            # If it's a callable (like te.autocast), call it to get a fresh context manager
            if callable(autocast_context):
                with autocast_context():
                    for _ in range(warmup_steps):
                        _ = model(x)
            else:
                with autocast_context:
                    for _ in range(warmup_steps):
                        _ = model(x)
        else:
            for _ in range(warmup_steps):
                _ = model(x)
    
    # Benchmark (CUDA required)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        if autocast_context is not None:
            # If it's a callable (like te.autocast), call it to get a fresh context manager
            if callable(autocast_context):
                with autocast_context():
                    for _ in range(steps):
                        x = model(x)
            else:
                with autocast_context:
                    for _ in range(steps):
                        x = model(x)
        else:
            for _ in range(steps):
                x = model(x)
    end.record()
    torch.cuda.synchronize()
    
    elapsed = start.elapsed_time(end)
    avg_time = elapsed / steps
    return avg_time


def run_benchmarks():
    """Run benchmarks for all transformer block implementations"""
    device = torch.device('cuda')
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}\n")
    
    # Common model parameters
    batch_size = 8
    seq_len = 512
    dim = 768
    num_heads = 12
    mlp_dim = 3072
    steps = 3
    
    results: Dict[str, float] = {}
    
    # 0. Baseline (FP32, NO Tensor Cores)
    try:
        torch.set_float32_matmul_precision('highest')
        model = BaselineTransformerBlock(dim, num_heads, mlp_dim).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        results['Baseline (FP32)'] = benchmark_model(model, x, steps, "Baseline (FP32)")
    except Exception as e:
        print(f"Error in Baseline: {e}")
    
    # 1. CuBLAS TF32 (FP32 weights, Tensor Core math)
    try:
        torch.set_float32_matmul_precision('high')
        model = CublasTransformerBlock(dim, num_heads, mlp_dim).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        results['CuBLAS TF32'] = benchmark_model(model, x, steps, "CuBLAS TF32")
    except Exception as e:
        print(f"Error in CuBLAS TF32: {e}")
    
    # 2. SDPA (Fused Kernels + FP32)
    # Note: SDPA may be slower than CuBLAS TF32 on short sequences because:
    try:
        torch.set_float32_matmul_precision('high')
        model = SDPATransformerBlock(dim, num_heads, mlp_dim, is_causal=False).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        results['SDPA (FP32)'] = benchmark_model(model, x, steps, "SDPA (FP32)")
    except Exception as e:
        print(f"Error in SDPA: {e}")
    
    # 3. Compiled (SDPA + Graph Fusion + FP32)
    try:
        torch.set_float32_matmul_precision('high')
        model = CompiledTransformerBlock(dim, num_heads, mlp_dim, is_causal=False).to(device)
        model = torch.compile(model, mode="max-autotune")
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        results['Compiled (FP32)'] = benchmark_model(model, x, steps, "Compiled (FP32)", warmup_steps=5)
    except Exception as e:
        print(f"Error in Compiled: {e}")
    
    # 4. Transformer Engine (FP8)
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    
    model = TransformerEngineBlock(dim, num_heads, mlp_dim).to(device)
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)
    
    autocast_ctx = lambda: te.autocast(enabled=True, recipe=fp8_recipe)
    results['Transformer Engine (FP8)'] = benchmark_model(
        model, x, steps, "Transformer Engine (FP8)", autocast_ctx
    )   
    
    # Print summary table
    if results:
        baseline_time = results.get('Baseline (FP32)', None)
        print("\n" + "=" * 70)
        print(f"{'Model':<35} {'ms/step':>10} {'Speedup':>10}")
        print("=" * 70)
        for name, time in results.items():
            if baseline_time and name != 'Baseline (FP32)':
                speedup = baseline_time / time
                print(f"{name:<35} {time:>10.2f} {speedup:>9.2f}x")
            else:
                print(f"{name:<35} {time:>10.2f} {'baseline':>10}")
        print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()
