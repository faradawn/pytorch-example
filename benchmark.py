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

# Transformer Engine (optional)
te_spec = importlib.util.spec_from_file_location("te", '4_transformer_engine.py')
te_module = importlib.util.module_from_spec(te_spec)
te_spec.loader.exec_module(te_module)
TE_AVAILABLE = getattr(te_module, 'TE_AVAILABLE', False)
TransformerEngineBlock = getattr(te_module, 'TransformerEngineBlock', None) if TE_AVAILABLE else None


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
        autocast_context: Optional autocast context manager
        warmup_steps: Number of warmup steps
    
    Returns:
        Average time per step in milliseconds
    """
    model.eval()
    device = x.device
    
    # Warmup
    with torch.no_grad():
        if autocast_context is not None:
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
    print(f"{name:40s}: {elapsed:7.2f} ms for {steps} steps ({avg_time:6.2f} ms/step)")
    return avg_time


def run_benchmarks():
    """Run benchmarks for all transformer block implementations"""
    device = torch.device('cuda')
    print(f"Running benchmarks on device: {device}")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    
    # Common model parameters
    batch_size = 8
    seq_len = 512
    dim = 768
    num_heads = 12
    mlp_dim = 3072
    steps = 10
    
    # Enable TF32 for NVIDIA Ampere+ (affects FP32 operations)
    torch.set_float32_matmul_precision('high')
    
    results: Dict[str, float] = {}
    
    # 0. Baseline (FP32)
    print("\n0. Baseline (FP32, Manual Attention)")
    print("-" * 80)
    try:
        model = BaselineTransformerBlock(dim, num_heads, mlp_dim).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        results['Baseline (FP32)'] = benchmark_model(model, x, steps, "Baseline (FP32)")
    except Exception as e:
        print(f"Error: {e}")
    
    # 1. CuBLAS TF32 (FP32 weights, Tensor Core math)
    print("\n1. CuBLAS TF32 (FP32 weights, Tensor Cores)")
    print("-" * 80)
    try:
        model = CublasTransformerBlock(dim, num_heads, mlp_dim).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
        results['CuBLAS TF32'] = benchmark_model(model, x, steps, "CuBLAS TF32")
    except Exception as e:
        print(f"Error: {e}")
    
    # 1b. CuBLAS FP16 (AMP, Tensor Cores)
    print("\n1b. CuBLAS FP16 (AMP, Tensor Cores)")
    print("-" * 80)
    try:
        model = CublasTransformerBlock(dim, num_heads, mlp_dim).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float16)
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        results['CuBLAS FP16'] = benchmark_model(model, x, steps, "CuBLAS FP16", autocast_ctx)
    except Exception as e:
        print(f"Error: {e}")
    
    # 2. SDPA (Fused Kernels + FP16)
    print("\n2. SDPA (Fused Kernels + FP16)")
    print("-" * 80)
    try:
        model = SDPATransformerBlock(dim, num_heads, mlp_dim, is_causal=False).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float16)
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        results['SDPA (FP16)'] = benchmark_model(model, x, steps, "SDPA (FP16)", autocast_ctx)
    except Exception as e:
        print(f"Error: {e}")
    
    # 3. Compiled (SDPA + Graph Fusion + FP16)
    print("\n3. Compiled (SDPA + Graph Fusion + FP16)")
    print("-" * 80)
    try:
        model = CompiledTransformerBlock(dim, num_heads, mlp_dim, is_causal=False).to(device)
        # Compile the model
        print("Compiling model (this may take a moment)...")
        model = torch.compile(model, mode="max-autotune")
        x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float16)
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        results['Compiled (FP16)'] = benchmark_model(model, x, steps, "Compiled (FP16)", autocast_ctx, warmup_steps=5)
    except Exception as e:
        print(f"Error: {e}")
    
    # 4. Transformer Engine (FP8)
    print("\n4. Transformer Engine (FP8)")
    print("-" * 80)
    if TE_AVAILABLE and TransformerEngineBlock is not None:
        try:
                import transformer_engine.pytorch as te
                from transformer_engine.common import recipe
                
                model = TransformerEngineBlock(dim, num_heads, mlp_dim).to(device)
                x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float16)
                
                # Create FP8 recipe
                fp8_recipe = recipe.DelayedScaling(
                    fp8_format=recipe.Format.E4M3,
                    margin=0,
                    interval=1,
                    fp8_amax_history_len=1024,
                    fp8_amax_compute_algo="most_recent",
                )
                
                autocast_ctx = te.autocast(enabled=True, recipe=fp8_recipe)
                results['Transformer Engine (FP8)'] = benchmark_model(
                    model, x, steps, "Transformer Engine (FP8)", autocast_ctx
                )
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Skipping: Transformer Engine not available")
        print("Install with: pip install transformer-engine[pytorch]")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if results:
        baseline_time = results.get('Baseline (FP32)', None)
        for name, time in results.items():
            if baseline_time:
                speedup = baseline_time / time
                print(f"{name:40s}: {time:6.2f} ms/step ({speedup:5.2f}x speedup vs baseline)")
            else:
                print(f"{name:40s}: {time:6.2f} ms/step")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmarks()
