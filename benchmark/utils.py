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


def save_results_to_csv(results):
    """Save forward and backward results to separate CSV files"""

    results_dir = "benchmark/results/model"
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
    forward_df.to_csv(os.path.join(results_dir, 'swin_transformer_v2_forward_comparaison.csv'), index=False)
    print("Forward pass results saved to 'swin_transformer_v2_forward_comparaison.csv'")
    
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
    backward_df.to_csv(os.path.join(results_dir, 'swin_transformer_v2_backward_comparaison.csv'), index=False)
    print("Backward pass results saved to 'swin_transformer_v2_backward_comparaison.csv'")
    
    return forward_df, backward_df


def create_plots(results):
    # Set up the plotting style
    results_dir = "benchmark/results/model"
    os.makedirs(results_dir, exist_ok=True)
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Swin Transformer V2 Performance: Flash vs Classic Attention', fontsize=16, fontweight='bold')
    
    batch_sizes = results['batch_sizes']
    
    # Color scheme
    classic_color = '#e74c3c'  # Red
    flash_color = '#3498db'    # Blue
    
    # Plot 1: Forward Pass - Execution Time
    ax1 = axes[0, 0]
    ax1.plot(batch_sizes, results['forward']['classic_time'], 'o-', 
             color=classic_color, linewidth=2, markersize=8, label='Classic Attention')
    ax1.plot(batch_sizes, results['forward']['flash_time'], 's-', 
             color=flash_color, linewidth=2, markersize=8, label='Flash Attention')
    ax1.set_title('Forward Pass - Execution Time', fontweight='bold')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Time (ms)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Forward Pass - Memory Usage
    ax2 = axes[0, 1]
    ax2.plot(batch_sizes, results['forward']['classic_memory'], 'o-', 
             color=classic_color, linewidth=2, markersize=8, label='Classic Attention')
    ax2.plot(batch_sizes, results['forward']['flash_memory'], 's-', 
             color=flash_color, linewidth=2, markersize=8, label='Flash Attention')
    ax2.set_title('Forward Pass - Memory Usage', fontweight='bold')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Memory (GB)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Backward Pass - Execution Time
    ax3 = axes[1, 0]
    ax3.plot(batch_sizes, results['backward']['classic_time'], 'o-', 
             color=classic_color, linewidth=2, markersize=8, label='Classic Attention')
    ax3.plot(batch_sizes, results['backward']['flash_time'], 's-', 
             color=flash_color, linewidth=2, markersize=8, label='Flash Attention')
    ax3.set_title('Backward Pass - Execution Time', fontweight='bold')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Time (ms)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Backward Pass - Memory Usage
    ax4 = axes[1, 1]
    ax4.plot(batch_sizes, results['backward']['classic_memory'], 'o-', 
             color=classic_color, linewidth=2, markersize=8, label='Classic Attention')
    ax4.plot(batch_sizes, results['backward']['flash_memory'], 's-', 
             color=flash_color, linewidth=2, markersize=8, label='Flash Attention')
    ax4.set_title('Backward Pass - Memory Usage', fontweight='bold')
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Memory (GB)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    
    plt.savefig(os.path.join(results_dir, 'swin_transformer_v2_flash_vs_classic.png'), dpi=300, bbox_inches='tight')
    plt.show()