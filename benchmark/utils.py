import torch

import torch
import time


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
    # For a vector of dimension D, 3D FLOPS (D mult, D-1 add, 1 sqrt, D div)
    # Total vectors to normalize: 2 * B * W * H * S
    flops_normalization = 2 * B * W * H * S * (3 * D)

    # 2. Dot Product of Q and K (Q @ K_T)
    # For (S, D) @ (D, S) -> (S, S) matrix multiplication:
    # S*S*D multiplications and S*S*(D-1) additions.
    # Total FLOPS for one (S,S) output: S^2 * D + S^2 * (D-1) = 2*S^2*D - S^2
    flops_dot_product = B * W * H * (2 * S**2 * D - S**2)

    # 3. Scaling by logit_scale and adding bias
    # Result of dot product is (B, W, H, S, S)
    # Each element of (S, S) matrix is multiplied by scalar and added by bias
    # S*S multiplications + S*S additions = 2*S^2
    flops_scaling_bias = B * W * H * (2 * S**2)

    # 4. Addition of the attention mask
    # Element-wise addition to the attention scores (S, S)
    # S*S additions = S^2
    flops_attention_mask = B * W * H * (S**2)

    # 5. Softmax
    # For a vector of length S: S exponentials, S-1 additions, S divisions.
    # Applied S times for each (S, S) matrix.
    # Total FLOPS for one (S, S) matrix: S * (S + (S-1) + S) = S * (3S - 1) = 3S^2 - S
    flops_softmax = B * W * H * (3 * S**2 - S)

    # 6. Matrix Multiplication with V (Attention_weights @ V)
    # Attention weights (S, S) @ V (S, D) -> (S, D)
    # S*D*S multiplications and S*D*(S-1) additions.
    # Total FLOPS for one (S, D) output: S^2 * D + S*D*(S-1) = 2*S^2*D - S*D
    flops_matmul_V = B * W * H * (2 * S**2 * D - S * D)

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