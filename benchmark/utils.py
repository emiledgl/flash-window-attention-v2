import os
import time

import torch
import pandas as pd
import matplotlib.pyplot as plt


def measure_speed_memory(f, *args, **kwargs):
    # warm up
    f(*args, **kwargs)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    for i in range(100):
        f(*args, **kwargs)
    torch.cuda.synchronize()
    t = time.time() - start
    memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

    return t, memory


def calculate_cosine_attention_flops(
    batch_size: int,
    nwindows: int,
    nheads: int,
    seqlen: int,
    head_dim: int,
) -> int:
    """
    Calculates the total number of floating-point operations (FLOPS)
    for the cosine attention mechanism.
    """

    B = batch_size
    W = nwindows
    H = nheads
    S = seqlen
    D = head_dim

    # 1. Normalization of Q and K
    flops_normalization = 2 * B * W * H * S * (3 * D)
    # 2. Dot Product of Q and K (Q @ K_T)
    flops_dot_product = B * W * H * (2 * S**2 * D)
    # 3. Scaling by logit_scale and adding bias
    flops_scaling_bias = B * W * H * (2 * S**2)
    # 4. Addition of the attention mask
    flops_attention_mask = B * W * H * (S**2)
    # 5. Softmax
    flops_softmax = B * W * H * (3 * S**2)
    # 6. Matrix Multiplication with V (Attention_weights @ V)
    flops_matmul_V = B * W * H * (2 * S**2 * D)

    # Summing all components
    total_flops = (
        flops_normalization +
        flops_dot_product +
        flops_scaling_bias +
        flops_attention_mask +
        flops_softmax +
        flops_matmul_V
    )
    
    return total_flops


def measure_memory_usage(func, *args, **kwargs):
    """
    Measure peak GPU memory usage for a function call.
    Returns peak memory in MB.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Get baseline memory
    baseline_memory = torch.cuda.memory_allocated()
    
    # Run the function
    result = func(*args, **kwargs)
    
    # Get peak memory usage
    peak_memory = torch.cuda.max_memory_allocated()
    memory_used = (peak_memory - baseline_memory) / (1024 ** 2)  # Convert to MB
    
    return result, memory_used


def save_results_to_csv(results, config):
    """Save forward and backward results to separate CSV files"""

    results_dir = "benchmark/results"
    os.makedirs(results_dir, exist_ok=True)

    # Forward pass results
    forward_data = {
        'batch_size': results['batch_sizes'],
        'classic_time_ms': results['forward']['classic_time'],
        'classic_memory_gb': results['forward']['classic_memory'],
        'flash_time_ms': results['forward']['flash_time'],
        'flash_memory_gb': results['forward']['flash_memory']
    }
    
    # Calculate speedup and memory ratios for forward pass
    forward_data['time_speedup'] = [c/f for c, f in zip(forward_data['classic_time_ms'], forward_data['flash_time_ms'])]
    forward_data['memory_ratio'] = [c/f for c, f in zip(forward_data['classic_memory_gb'], forward_data['flash_memory_gb'])]
    
    forward_df = pd.DataFrame(forward_data)
    forward_df.to_csv(os.path.join(results_dir, f'swin_transformer_v2_forward_comparaison-patch{config["patch_size"]}-window{config["window_size"]}.csv'), index=False)
    
    # Backward pass results
    backward_data = {
        'batch_size': results['batch_sizes'],
        'classic_time_ms': results['backward']['classic_time'],
        'classic_memory_gb': results['backward']['classic_memory'],
        'flash_time_ms': results['backward']['flash_time'],
        'flash_memory_gb': results['backward']['flash_memory']
    }
    
    # Calculate speedup and memory ratios for backward pass
    backward_data['time_speedup'] = [c/f for c, f in zip(backward_data['classic_time_ms'], backward_data['flash_time_ms'])]
    backward_data['memory_ratio'] = [c/f for c, f in zip(backward_data['classic_memory_gb'], backward_data['flash_memory_gb'])]
    
    backward_df = pd.DataFrame(backward_data)
    backward_df.to_csv(os.path.join(results_dir, f'swin_transformer_v2_backward_comparaison-patch{config["patch_size"]}-window{config["window_size"]}.csv'), index=False)
    
    return forward_df, backward_df