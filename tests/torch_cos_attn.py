import torch
import torch.nn.functional as F

class CosAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, logit_scale, bias=None, mask=None):
        # Normalize q and k for cosine similarity
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)

        # Calculate raw attention scores (cosine similarity)
        # (B, H, N, Dk) @ (B, H, Dk, M) -> (B, H, N, M)
        attn_scores_raw = torch.matmul(q_norm, k_norm.transpose(-2, -1))

        # Apply logit scale. Clamp and then exponentiate.
        # The clamp value 4.6052 is approximately log(100), limiting the scale to 100.
        effective_logit_scale = logit_scale.clamp(max=4.6052).exp()
        attn_scores_scaled = attn_scores_raw * effective_logit_scale

        # Add bias if provided
        if bias is not None:
            attn_scores_scaled += bias

        # Add mask if provided (typically for padding or causality).
        # Mask values should be very small (e.g., -inf) for positions to ignore.
        if mask is not None:
            attn_scores_scaled += mask

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores_scaled, dim=-1)

        # Calculate output: attention weights @ V
        # (B, H, N, M) @ (B, H, M, Dv) -> (B, H, N, Dv)
        output = torch.matmul(attn_weights, v)

        # Save tensors for backward pass
        ctx.save_for_backward(q, k, v, logit_scale, bias, attn_weights, attn_scores_raw, q_norm, k_norm)
        
        # Store original bias shape for gradient summation if broadcasting occurred
        ctx.bias_shape = bias.shape if bias is not None else None

        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, logit_scale, bias, attn_weights, attn_scores_raw, q_norm, k_norm = ctx.saved_tensors
        bias_shape = ctx.bias_shape

        dq = dk = dv = d_logit_scale = dbias = None

        # 1. Gradient with respect to V (dL/dV)
        # O = P @ V  => dL/dV = P.transpose(-2, -1) @ dL/dO
        dv = torch.matmul(attn_weights.transpose(-2, -1), grad_output)

        # 2. Gradient with respect to Attention Weights (dL/dP)
        # O = P @ V  => dL/dP = dL/dO @ V.transpose(-2, -1)
        grad_attn_weights = torch.matmul(grad_output, v.transpose(-2, -1))

        # 3. Gradient through Softmax (dL/d_attn_scores_scaled)
        # This is the numerically stable softmax backward formula: P * (dL/dP - sum(dL/dP * P, dim=-1, keepdim=True))
        grad_attn_scores_scaled = attn_weights * (grad_attn_weights - (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True))

        # 4. Gradient with respect to Bias (dL/dbias)
        if bias is not None:
            dbias = grad_attn_scores_scaled
            # If bias was broadcasted, sum gradients over the broadcasted dimensions
            if bias_shape is not None and len(bias_shape) < len(dbias.shape):
                # Calculate dimensions to sum over (e.g., if bias is (N, M) and dbias is (B, H, N, M), sum over B and H)
                dims_to_sum = tuple(range(len(dbias.shape) - len(bias_shape)))
                dbias = dbias.sum(dim=dims_to_sum, keepdim=False)
                dbias = dbias.reshape(bias_shape) # Ensure the gradient matches original bias shape

        # 5. Gradient with respect to logit_scale (dL/d_logit_scale)
        # effective_logit_scale = logit_scale.clamp(max=4.6052).exp()
        # attn_scores_scaled = attn_scores_raw * effective_logit_scale

        # dL/d(effective_logit_scale) = grad_attn_scores_scaled * attn_scores_raw
        grad_effective_logit_scale = (grad_attn_scores_scaled * attn_scores_raw).sum(dim=(0, 2, 3), keepdim=True)
        
        # d(effective_logit_scale)/d(logit_scale) = exp(logit_scale) if logit_scale < 4.6052 else 0
        effective_logit_scale_val = logit_scale.clamp(max=4.6052).exp()
        
        # Apply chain rule: dL/d(logit_scale) = dL/d(effective_logit_scale) * d(effective_logit_scale)/d(logit_scale)
        d_logit_scale = grad_effective_logit_scale * effective_logit_scale_val
        
        # Apply the derivative of the clamp function (Heaviside step function)
        # If logit_scale was clamped, its gradient is 0.
        clamp_mask = (logit_scale < 4.6052).float()
        d_logit_scale = d_logit_scale * clamp_mask

        # 6. Gradient with respect to Raw Attention Scores (dL/d_attn_scores_raw)
        # grad_attn_scores_scaled is dL/d(attn_scores_raw * effective_logit_scale)
        grad_attn_scores_raw = grad_attn_scores_scaled * effective_logit_scale_val

        # 7. Gradients with respect to Q and K (dL/dQ, dL/dK)
        # attn_scores_raw = Q_norm @ K_norm.transpose(-2, -1)
        # We need gradients with respect to q_norm and k_norm first.
        
        # dL/dQ_norm = dL/d_attn_scores_raw @ K_norm
        dq_norm_pre_norm = torch.matmul(grad_attn_scores_raw, k_norm)
        
        # dL/dK_norm_T = Q_norm.transpose(-2, -1) @ dL/d_attn_scores_raw
        # dL/dK_norm = (dL/dK_norm_T).transpose(-2, -1)
        dk_norm_pre_norm = torch.matmul(q_norm.transpose(-2, -1), grad_attn_scores_raw).transpose(-2, -1)

        # Gradients through L2 normalization:
        # For y = x / ||x||, dy/dx = (I * ||x||^2 - x * x.T) / ||x||^3
        # Which simplifies to (I - y * y.T) / ||x||
        
        # dq = (I - q_norm @ q_norm.transpose(-2, -1)) / norm_q * dq_norm_pre_norm
        # More stably: dq = dq_norm_pre_norm / norm_q - (q_norm * (dq_norm_pre_norm * q_norm).sum(dim=-1, keepdim=True)) / norm_q
        
        norm_q = q.norm(p=2, dim=-1, keepdim=True)
        norm_k = k.norm(p=2, dim=-1, keepdim=True)

        dq = dq_norm_pre_norm / norm_q - (q * (dq_norm_pre_norm * q_norm).sum(dim=-1, keepdim=True)) / (norm_q**2)
        dk = dk_norm_pre_norm / norm_k - (k * (dk_norm_pre_norm * k_norm).sum(dim=-1, keepdim=True)) / (norm_k**2)

        # Return gradients in the same order as inputs to forward (excluding ctx)
        # q, k, v, logit_scale, bias, mask
        return dq, dk, dv, d_logit_scale, dbias, None


# Helper function to use the custom attention function
def cos_attention(q, k, v, scale, bias=None, mask=None):
    return CosAttentionFunction.apply(q, k, v, scale, bias, mask)

# Example Usage:
if __name__ == '__main__':
    # Define dimensions
    batch_size = 4
    num_heads = 8
    query_seq_len = 32
    key_seq_len = 32
    head_dim = 64

    # Create random input tensors
    q = torch.randn(batch_size, num_heads, query_seq_len, head_dim, requires_grad=True)
    k = torch.randn(batch_size, num_heads, key_seq_len, head_dim, requires_grad=True)
    v = torch.randn(batch_size, num_heads, key_seq_len, head_dim, requires_grad=True)

    # Optional: Bias and Mask
    # Bias can be broadcasted, e.g., (1, num_heads, query_seq_len, key_seq_len)
    bias = torch.randn(1, num_heads, query_seq_len, key_seq_len, requires_grad=True)
    
    # Mask for padding (example: mask out last 2 key positions for all queries)
    # Mask should be additive, so use large negative values for masked positions.
    mask = torch.zeros(1, 1, query_seq_len, key_seq_len)
    mask[:, :, :, key_seq_len-2:] = -torch.inf # Example: mask out last 2 key positions
    mask = mask.to(q.device) # Ensure mask is on the same device as q

    # logit_scale should be a tensor with requires_grad=True
    logit_scale = torch.nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)


    output = cos_attention(q, k, v, logit_scale, bias=bias, mask=mask)
    loss = output.sum()
    loss.backward()

    print("\n--- Gradients from Custom Attention Function (qk_scale as tensor requiring grad) ---")
    print(f"Gradient for q (dq) shape: {q.grad.shape}")
    print(f"Gradient for k (dk) shape: {k.grad.shape}")
    print(f"Gradient for v (dv) shape: {v.grad.shape}")
    if bias is not None:
        print(f"Gradient for bias (dbias) shape: {bias.grad.shape}")
    print(f"Gradient for logit_scale (d_logit_scale) shape: {logit_scale.grad.shape}") # Should be a tensor for tensor input

    # --- Verification with PyTorch's built-in attention (for comparison) ---
    print("\n--- Verifying with PyTorch's Built-in Operations ---")
    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    bias_pt = bias.detach().clone().requires_grad_(True) if bias is not None else None
    logit_scale_pt = logit_scale.detach().clone().requires_grad_(True)


    # Calculate attention scores
    q_pt_norm = q_pt / torch.norm(q_pt, dim=-1, p=2, keepdim=True)
    k_pt_norm = k_pt / torch.norm(k_pt, dim=-1, p=2, keepdim=True)
    attn_scores_raw_pt = torch.matmul(q_pt_norm, k_pt_norm.transpose(-2, -1))
    
    effective_logit_scale_pt = logit_scale_pt.clamp(max=4.6052).exp()
    attn_scores_scaled_pt = attn_scores_raw_pt * effective_logit_scale_pt

    if bias_pt is not None:
        attn_scores_biased_pt = attn_scores_scaled_pt + bias_pt
    else:
        attn_scores_biased_pt = attn_scores_scaled_pt

    if mask is not None:
        attn_scores_masked_pt = attn_scores_biased_pt + mask
    else:
        attn_scores_masked_pt = attn_scores_biased_pt

    attn_weights_pt = F.softmax(attn_scores_masked_pt, dim=-1)
    output_pt = torch.matmul(attn_weights_pt, v_pt)

    loss_pt = output_pt.sum()
    loss_pt.backward()

    print("\n--- Gradients from PyTorch's Built-in Operations ---")
    print(f"Gradient for q (dq_pt) shape: {q_pt.grad.shape}")
    print(f"Gradient for k (dk_pt) shape: {k_pt.grad.shape}")
    print(f"Gradient for v (dv_pt) shape: {v_pt.grad.shape}")
    if bias_pt is not None:
        print(f"Gradient for bias (dbias_pt) shape: {bias_pt.grad.shape}")
    print(f"Gradient for logit_scale (d_logit_scale_pt) shape: {logit_scale_pt.grad.shape}")

    # Compare outputs and gradients (should be very close)
    print("\n--- Comparison ---")
    print(f"Output close: {torch.allclose(output, output_pt, rtol=1e-4, atol=1e-4)}")
    print(f"dq close: {torch.allclose(q.grad, q_pt.grad, rtol=1e-4, atol=1e-4)}")
    print(f"dk close: {torch.allclose(k.grad, k_pt.grad, rtol=1e-4, atol=1e-4)}")
    print(f"dv close: {torch.allclose(v.grad, v_pt.grad, rtol=1e-4, atol=1e-4)}")
    if bias is not None:
        print(f"dbias close: {torch.allclose(bias.grad, bias_pt.grad, rtol=1e-4, atol=1e-4)}")
    print(f"logit_scale close: {torch.allclose(logit_scale.grad, logit_scale_pt.grad, rtol=1e-4, atol=1e-4)}")

     # --- Absolute & Relative Differences ---
    print("\n--- Absolute Differences ---")
    output_diff = torch.abs(output - output_pt)
    print(f"Output max absolute difference: {output_diff.max().item():.2e}")
    output_rel_diff = (output_diff / (torch.abs(output_pt) + 1e-10)) * 100
    print(f"Output max relative difference: {output_rel_diff.max().item():.6f}%")
    
    dq_diff = torch.abs(q.grad - q_pt.grad)
    print(f"dq max absolute difference: {dq_diff.max().item():.2e}")
    dq_rel_diff = (dq_diff / (torch.abs(q_pt.grad) + 1e-10)) * 100
    print(f"dq max relative difference: {dq_rel_diff.max().item():.6f}%")
    
    dk_diff = torch.abs(k.grad - k_pt.grad)
    print(f"dk max absolute difference: {dk_diff.max().item():.2e}")
    dk_rel_diff = (dk_diff / (torch.abs(k_pt.grad) + 1e-10)) * 100
    print(f"dk max relative difference: {dk_rel_diff.max().item():.6f}%")
    
    dv_diff = torch.abs(v.grad - v_pt.grad)
    print(f"dv max absolute difference: {dv_diff.max().item():.2e}")
    dv_rel_diff = (dv_diff / (torch.abs(v_pt.grad) + 1e-10)) * 100
    print(f"dv max relative difference: {dv_rel_diff.max().item():.6f}%")
    
    if bias is not None:
        dbias_diff = torch.abs(bias.grad - bias_pt.grad)
        print(f"dbias max absolute difference: {dbias_diff.max().item():.2e}")
        dbias_rel_diff = (dbias_diff / (torch.abs(bias_pt.grad) + 1e-10)) * 100
        print(f"dbias max relative difference: {dbias_rel_diff.max().item():.2f}%")
    
    dlogit_scale_diff = torch.abs(logit_scale.grad - logit_scale_pt.grad)
    print(f"logit_scale max absolute difference: {dlogit_scale_diff.max().item():.2e}")
    dlogit_scale_rel_diff = (dlogit_scale_diff / (torch.abs(logit_scale_pt.grad) + 1e-10)) * 100
    print(f"logit_scale max relative difference: {dlogit_scale_rel_diff.max().item():.6f}%")