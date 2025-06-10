import torch
import triton
from .kernels import _fwd_kernel, _bwd_kernel

HEAD_CHUNK_DIM = 16

def flash_swin_attn_fwd_func(Q, K, V, bias, mask, scale_qk):
    batch, heads, n_windows, seq, head_dim = Q.size()
    seq_pad = triton.next_power_of_2(seq)
    head_chunk = head_dim // HEAD_CHUNK_DIM
    O = torch.empty_like(Q)

    grid = (batch, heads, n_windows)
    _fwd_kernel[grid](
        Q,
        K,
        V,
        bias,
        mask,
        O,
        scale_qk,
        n_windows,
        heads,
        head_dim,
        head_chunk,
        HEAD_CHUNK_DIM,
        seq,
        seq_pad,
    )

    return O

def flash_swin_attn_bwd_func(Q, K, V, bias, mask, d_O, scale_qk):
    batch, heads, n_windows, seq, head_dim = Q.size()
    seq_pad = triton.next_power_of_2(seq)
    head_chunk = head_dim // HEAD_CHUNK_DIM

    d_Q = torch.empty_like(Q)
    d_K = torch.empty_like(K)
    d_V = torch.empty_like(V)
    d_bias = torch.zeros_like(bias)

    grid = (batch, heads, n_windows)
    _bwd_kernel[grid](
        Q,
        K,
        V,
        bias,
        mask,
        d_O,
        d_Q,
        d_K,
        d_V,
        d_bias,
        scale_qk,
        n_windows,
        heads,
        head_dim,
        head_chunk,
        HEAD_CHUNK_DIM,
        seq,
        seq_pad,
    )

    return d_Q, d_K, d_V, d_bias

class FlashSwinFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, mask, scale_qk):
        assert q.size(-1) & 15 == 0, "head_dim must be divisible by 16"
        o = flash_swin_attn_fwd_func(q, k, v, bias, mask, scale_qk)
        ctx.save_for_backward(q, k, v, bias, mask)
        ctx.scale_qk = scale_qk
        return o

    @staticmethod
    def backward(ctx, d_o):
        q, k, v, bias, mask = ctx.saved_tensors
        d_q, d_k, d_v, d_bias = flash_swin_attn_bwd_func(q, k, v, bias, mask, d_o, ctx.scale_qk)

        return d_q, d_k, d_v, d_bias, None, None


flash_swin_attn = FlashSwinFunc.apply