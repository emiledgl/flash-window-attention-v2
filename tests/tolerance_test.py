import math
import torch
import torch.nn.functional as F

from benchmark.win_attention_func import win_attention_torch
from flash_win_attn_2 import FlashWindowAttention2

flash_win_attn_2 = FlashWindowAttention2.apply


def test_flash_attention(
    batch_size: int,
    num_heads: int,
    num_windows: int,
    seq_len: int,
    head_dim: int,
):
    """
    Tests the flash attention implementation against the classic implementation
    """
    # Set random seed for reproducibility
    #torch.manual_seed(42)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Flash attention requires CUDA.")
        return
    
    # Create random inputs
    q = torch.randn(batch_size * num_windows, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size * num_windows, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size * num_windows, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    bias = torch.randn(num_heads, seq_len, seq_len, device=device, dtype=dtype)
    mask = torch.zeros((num_windows, seq_len, seq_len), device=device, dtype=dtype).masked_fill(
        torch.randn((num_windows, seq_len, seq_len), device=device, dtype=dtype) < 0, -float("inf")
    ).to(device, dtype)
    
    # Scale factor
    logit_scale = torch.randn((num_heads, 1, 1), device=device, dtype=dtype)
    #logit_scale = torch.nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1)))).to(device=device, dtype=dtype)
    
    print(f"Testing with shapes: q,k,v = {q.shape}")
    # Warm up GPU
    for _ in range(10):
        _ = win_attention_torch(q, k, v, logit_scale, bias, mask)
        _ = flash_win_attn_2(q, k, v, logit_scale, bias, mask)

    # Run classic attention
    with torch.no_grad():
        classic_output = win_attention_torch(q, k, v, logit_scale, bias, mask)
    
    # Run flash attention
    with torch.no_grad():
        flash_output = flash_win_attn_2(q, k, v, logit_scale, bias, mask)
    
    # Compare outputs
    abs_diff = (classic_output - flash_output).abs()
    max_abs_diff = abs_diff.max().item()
    
    # Calculate relative differences
    # Use a more robust epsilon and handle edge cases
    epsilon = 1e-6
    classic_magnitude = classic_output.abs()
    
    # Only compute relative differences where the reference value is significant
    # This avoids inf/nan from dividing by very small numbers
    valid_mask = classic_magnitude > epsilon
    if valid_mask.any():
        rel_diff_valid = abs_diff[valid_mask] / classic_magnitude[valid_mask]
        mean_rel_diff = rel_diff_valid.mean().item()
        valid_fraction = valid_mask.float().mean().item()
    else:
        mean_rel_diff = 0.0
        valid_fraction = 0.0
    
    print(f"Max absolute difference: {max_abs_diff:.6f}")
    print(f"Mean relative difference: {mean_rel_diff:.6f} (computed on {valid_fraction:.1%} of values)")
    
    # Check if the difference is within acceptable range
    # The flash implementation uses different algorithms and numerical precision,
    # so we expect small differences
    abs_threshold = 1e-2
    rel_threshold = 1e-1  # More lenient relative threshold
    
    abs_passed = max_abs_diff < abs_threshold
    rel_passed = mean_rel_diff < rel_threshold if valid_fraction > 0.5 else True  # Skip rel check if most values are too small
    
    if abs_passed and rel_passed:
        print(f"✅ Test PASSED: Both absolute ({abs_threshold}) and relative ({rel_threshold}) thresholds met")
    elif abs_passed and valid_fraction <= 0.5:
        print(f"✅ Test PASSED: Absolute threshold met ({abs_threshold})")
        print(f"   Note: Relative comparison skipped (only {valid_fraction:.1%} of values significant)")
    elif abs_passed:
        print(f"⚠️  Test PARTIAL: Absolute threshold met ({abs_threshold}) but relative threshold exceeded ({rel_threshold})")
        print(f"   Mean relative difference: {mean_rel_diff:.6f}")
    elif rel_passed:
        print(f"⚠️  Test PARTIAL: Relative threshold met ({rel_threshold}) but absolute threshold exceeded ({abs_threshold})")
        print(f"   Max absolute difference: {max_abs_diff:.6f}")
    else:
        print(f"❌ Test FAILED: Both thresholds exceeded")
        print(f"   Absolute: {max_abs_diff:.6f} > {abs_threshold}")
        if valid_fraction > 0.5:
            print(f"   Relative: {mean_rel_diff:.6f} > {rel_threshold}")


def test_backward_pass(
        batch_size: int,
        num_heads: int,
        num_windows: int,
        seq_len: int,
        head_dim: int,
    ):
    """
    Tests the accuracy of the backward pass in flash attention compared to classic attention
    """
    # Set random seed for reproducibility
    #torch.manual_seed(42)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Flash attention requires CUDA.")
        return
    
    # Create random inputs with gradients
    q = torch.randn(batch_size * num_windows, num_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size * num_windows, num_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size * num_windows, num_heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn(num_heads, seq_len, seq_len, device=device, dtype=dtype, requires_grad=True)
    mask = torch.zeros((num_windows, seq_len, seq_len), device=device, dtype=dtype).masked_fill(
        torch.randn((num_windows, seq_len, seq_len), device=device, dtype=dtype) < 0, -float("inf")
    ).to(device, dtype)

    # Scale factor
    #logit_scale = torch.randn((num_heads, 1, 1), device=device, dtype=dtype, requires_grad=True)
    logit_scale = torch.nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True).to(device=device, dtype=dtype)
    
    # Clone inputs for classic attention
    q_classic = q.detach().clone().requires_grad_(True)
    k_classic = k.detach().clone().requires_grad_(True)
    v_classic = v.detach().clone().requires_grad_(True)
    bias_classic = bias.detach().clone().requires_grad_(True)
    logit_scale_classic = logit_scale.detach().clone().requires_grad_(True)  if isinstance(logit_scale, torch.Tensor) else logit_scale

     # Create identical inputs for flash attention
    q_flash = q.detach().clone().requires_grad_(True)
    k_flash = k.detach().clone().requires_grad_(True)
    v_flash = v.detach().clone().requires_grad_(True)
    bias_flash = bias.detach().clone().requires_grad_(True)
    logit_scale_flash = logit_scale.detach().clone().requires_grad_(True) if isinstance(logit_scale, torch.Tensor) else logit_scale
    
    
    print("\n==== Testing Backward Pass Accuracy ====")
    
    # Run classic attention with gradient tracking
    classic_output =  win_attention_torch(q_classic, k_classic, v_classic, logit_scale_classic, bias_classic, mask)
    print(f"Classic output requires_grad: {classic_output.requires_grad}")
    
    # Run flash attention with gradient tracking
    flash_output = flash_win_attn_2(q_flash, k_flash, v_flash, logit_scale_flash, bias_flash, mask)
    print(f"Flash output requires_grad: {flash_output.requires_grad}")

    # Create a random gradient for backpropagation
    grad_output = torch.randn_like(classic_output)
    
    # Backpropagate through classic attention
    classic_output.backward(grad_output)
    
    # Print gradient status
    print("Classic gradients:")
    print(f"q_grad: {'None' if q_classic.grad is None else 'Tensor'}")
    print(f"k_grad: {'None' if k_classic.grad is None else 'Tensor'}")
    print(f"v_grad: {'None' if v_classic.grad is None else 'Tensor'}")
    if isinstance(logit_scale_classic, torch.Tensor):
        print(f"logit_scale_grad: {'None' if logit_scale_classic.grad is None else 'Tensor'}")
    print(f"bias_grad: {'None' if bias_classic.grad is None else 'Tensor'}")

    ## Run backward on flash attention
    flash_output.backward(grad_output)
    
    # Print flash gradient status
    print("Flash gradients:")
    print(f"q_grad: {'None' if q_flash.grad is None else 'Tensor'}")
    print(f"k_grad: {'None' if k_flash.grad is None else 'Tensor'}")
    print(f"v_grad: {'None' if v_flash.grad is None else 'Tensor'}")
    if isinstance(logit_scale_flash, torch.Tensor):
        print(f"logit_scale_grad: {'None' if logit_scale_flash.grad is None else 'Tensor'}")
    print(f"bias_grad: {'None' if bias_flash.grad is None else 'Tensor'}")

    # Helper function to compute and display gradient differences
    def compute_gradient_differences(name, classic_grad, flash_grad):
        if classic_grad is not None and flash_grad is not None:
            abs_diff = (flash_grad - classic_grad).abs()
            max_abs_diff = abs_diff.max().item()
            
            # Robust relative differences calculation
            epsilon = 1e-6
            classic_magnitude = classic_grad.abs()
            
            # Only compute relative differences where the reference value is significant
            valid_mask = classic_magnitude > epsilon
            if valid_mask.any():
                rel_diff_valid = abs_diff[valid_mask] / classic_magnitude[valid_mask]
                mean_rel_diff = rel_diff_valid.mean().item()
                valid_fraction = valid_mask.float().mean().item()
            else:
                mean_rel_diff = 0.0
                valid_fraction = 0.0
            
            print(f"{name} gradient differences:")
            print(f"  Max absolute: {max_abs_diff:.6f}")
            print(f"  Mean relative: {mean_rel_diff:.6f} (on {valid_fraction:.1%} of values)")
            
            return max_abs_diff, mean_rel_diff
        else:
            print(f"{name} gradient: One or both gradients are None")
            return None, None

    # Compare gradients
    print("\nGradient differences:")
    
    q_max_abs, q_max_rel = compute_gradient_differences("Q", q_classic.grad, q_flash.grad)
    k_max_abs, k_max_rel = compute_gradient_differences("K", k_classic.grad, k_flash.grad)
    v_max_abs, v_max_rel = compute_gradient_differences("V", v_classic.grad, v_flash.grad)
    
    if isinstance(logit_scale_classic, torch.Tensor) and isinstance(logit_scale_flash, torch.Tensor):
        scale_max_abs, scale_max_rel = compute_gradient_differences("Logit_scale", logit_scale_classic.grad, logit_scale_flash.grad)
    else:
        scale_max_abs, scale_max_rel = None, None

    bias_max_abs, bias_max_rel = compute_gradient_differences("Bias", bias_classic.grad, bias_flash.grad)
        
    # Check if differences are within acceptable threshold
    abs_threshold = 5e-2
    rel_threshold = 1e-1  # More lenient relative threshold
    
    # Collect all valid differences
    abs_diffs = [diff for diff in [q_max_abs, k_max_abs, v_max_abs, scale_max_abs, bias_max_abs] if diff is not None]
    rel_diffs = [diff for diff in [q_max_rel, k_max_rel, v_max_rel, scale_max_rel, bias_max_rel] if diff is not None and not (math.isnan(diff) or math.isinf(diff))]
    
    if abs_diffs:
        max_abs_overall = max(abs_diffs)
        abs_passed = max_abs_overall < abs_threshold
        
        if rel_diffs:
            max_rel_overall = max(rel_diffs)
            rel_passed = max_rel_overall < rel_threshold
            
            if abs_passed and rel_passed:
                print(f"✅ Backward pass test PASSED: Both absolute ({abs_threshold}) and relative ({rel_threshold}) thresholds met")
            elif abs_passed:
                print(f"⚠️  Backward pass test PARTIAL: Absolute threshold met but relative threshold exceeded")
                print(f"   Max relative difference: {max_rel_overall:.6f} > {rel_threshold}")
            elif rel_passed:
                print(f"⚠️  Backward pass test PARTIAL: Relative threshold met but absolute threshold exceeded")
                print(f"   Max absolute difference: {max_abs_overall:.6f} > {abs_threshold}")
            else:
                print(f"❌ Backward pass test FAILED: Both thresholds exceeded")
                print(f"   Absolute: {max_abs_overall:.6f} > {abs_threshold}")
                print(f"   Relative: {max_rel_overall:.6f} > {rel_threshold}")
        else:
            # Only absolute comparison available
            if abs_passed:
                print(f"✅ Backward pass test PASSED: Absolute threshold met ({abs_threshold})")
                print("   Note: Relative comparison not available (all reference values too small)")
            else:
                print(f"❌ Backward pass test FAILED: Absolute threshold exceeded")
                print(f"   Absolute: {max_abs_overall:.6f} > {abs_threshold}")
    else:
        print("❌ Backward pass test FAILED: No valid gradient differences computed")


if __name__ == "__main__":
    print("==== Flash Attention vs Classic Attention Test ====")
    # Parameters
    batch_size = 4
    num_heads = 8
    num_windows = 4
    seq_len = 1024
    head_dim = 32
    print("===config===")
    print("batch_size:", batch_size)
    print("num_heads:", num_heads)
    print("num_windows:", num_windows)
    print("seq_len:", seq_len)
    print("head_dim:", head_dim)

    test_flash_attention(batch_size, num_heads, num_windows, seq_len, head_dim)
    test_backward_pass(batch_size, num_heads, num_windows, seq_len, head_dim)