import math
import torch
import triton
import triton.language as tl


# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
#         triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
#         triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=1),
#     ],
#     key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BLOCK_HEADDIM']
# )
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Scale,
    Bias,
    Mask,
    O,
    Lse,
    stride_qbw,
    stride_qh,
    stride_qm,
    stride_kbw,
    stride_kh,
    stride_kn,
    stride_vbw,
    stride_vh,
    stride_vn,
    stride_bh,
    stride_bm,
    stride_mw,
    stride_mm,
    stride_lse_bw,
    stride_lse_h,
    stride_obw,
    stride_oh,
    stride_om,
    n_windows,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    SCALE_TYPE: tl.constexpr, # "scalar" or "tensor"
    HAS_BIAS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bw = tl.program_id(1)
    off_h = tl.program_id(2)
    LOG2E: tl.constexpr = tl.log2(tl.exp(1.0))  # log2(e)
    MAX_LOG: tl.constexpr = tl.log(100.0)

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    
    # Initialize pointers to Q, K, V
    q_ptrs = (
        Q + off_bw * stride_qbw + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_bw * stride_kbw + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_bw * stride_vbw + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    # Initialize pointers to Bias and Mask
    if HAS_BIAS:
        bias_ptrs = (
            Bias
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    if HAS_MASK:
        off_w = off_bw % n_windows
        mask_ptrs = (
            Mask
            + off_w * stride_mw
            + (offs_m[:, None] * stride_mm + offs_n[None, :])
        )
    
    # Initialize accumulators and tracking variables
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    
    # Scale for softmax
    scale_ptr = Scale + off_h if SCALE_TYPE == "tensor" else Scale
    log_scale = tl.load(scale_ptr).to(tl.float32)
    if SCALE_TYPE == "tensor":
        logit_scale = tl.minimum(log_scale, MAX_LOG).exp()
    else:
        logit_scale = log_scale
    qk_scale = logit_scale * LOG2E
    
    # load q: it will stay in SRAM throughout
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )
    norm_q = tl.sqrt(tl.sum(q.to(tl.float32) * q.to(tl.float32), axis=1, keep_dims=True) + 1e-12)
    q_n = q.to(tl.float32) / norm_q
    q_n = (q_n * qk_scale).to(tl.float16)
    
    # Loop over k, v and update accumulator
    for start_n in range(0, seqlen_k, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # Calculate mask for valid positions in this key/value block
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        norm_k = tl.sqrt(tl.sum(k.to(tl.float32) * k.to(tl.float32), axis=1, keep_dims=True) + 1e-12)
        k_n = k.to(tl.float32) / norm_k
        k_n = k_n.to(tl.float16)

        # Compute qk with attention masking
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q_n, tl.trans(k_n))

        if HAS_BIAS:
            if EVEN_M & EVEN_N:
                bias = tl.load(bias_ptrs + start_n).to(tl.float32)
            else:
                bias = tl.load(
                    bias_ptrs + start_n,
                    mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                    other=0.0,
                ).to(tl.float32)
            bias *= LOG2E
            qk += bias
        if HAS_MASK:
            if EVEN_M & EVEN_N:
                mask = tl.load(mask_ptrs + start_n).to(tl.float32)
            else:
                mask = tl.load(
                    mask_ptrs + start_n,
                    mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                    other=float("-inf"),
                ).to(tl.float32)
            qk += mask
        else:
            if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
                qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        
        # Compute scaling constant
        m_ij = tl.maximum(tl.max(qk, 1), m_i)
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        
        # Scale and update accumulator
        acc_scale = lse_i * 0 + alpha  # Workaround for compiler bug
        acc_o *= acc_scale[:, None]

        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        
        # Update m_i and l_i
        m_i = m_ij
        lse_i = lse_i * alpha + tl.sum(p, 1)
    
    # Final normalization
    acc_o = tl.where(lse_i[:, None] > 0, acc_o / lse_i[:, None], 0.0)
    
    # write back l and m
    lse_ptrs = Lse + off_bw * stride_lse_bw + off_h * stride_lse_h + offs_m
    tl.store(lse_ptrs, m_i + tl.math.log2(lse_i))
    
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
            O
            + off_bw * stride_obw
            + off_h * stride_oh
            + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )


def _flash_attn_forward(q, k, v, logit_scale, bias=None, mask=None, scale_type="scalar"):

        batch_window_size, n_heads, seqlen_q, d = q.shape
        _, _, seqlen_k, _ = k.shape
        assert k.shape == (batch_window_size, n_heads, seqlen_k, d)
        assert v.shape == (batch_window_size, n_heads, seqlen_k, d)
        assert d <= 128
        assert q.dtype == k.dtype == v.dtype
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and k.is_cuda and v.is_cuda and logit_scale.is_cuda

        if scale_type == "tensor":
            assert logit_scale.shape == (n_heads, 1, 1)
            assert logit_scale.dtype == q.dtype
        
        has_bias = bias is not None
        if has_bias:
            assert bias.dtype == q.dtype
            assert bias.is_cuda
            assert bias.shape == (n_heads, seqlen_q, seqlen_k)
            if bias.stride(-1) != 1:
                bias = bias.contiguous()
        bias_strides = (bias.stride(0), bias.stride(1)) if has_bias else (0, 0)

        has_mask = mask is not None
        if has_mask:
            assert mask.dtype in [q.dtype, torch.float]
            assert mask.is_cuda
            if mask.stride(-1) != 1:
                mask = mask.contiguous()
        n_windows = mask.shape[0] if mask is not None else None
        mask_strides = (mask.stride(0), mask.stride(1)) if has_mask else (0, 0)

        seqlen_q_rounded = math.ceil(seqlen_q / 64) * 64
        lse = torch.empty((batch_window_size, n_heads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
        o = torch.empty_like(q)

        BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
        BLOCK_M = 32 if seqlen_q <= 64 else 64
        BLOCK_N = 32 if seqlen_k <= 64 else 64
        num_warps = 4 if d <= 64 else 8
        grid = (triton.cdiv(seqlen_q, BLOCK_M), batch_window_size, n_heads)

        _fwd_kernel[grid](
            q,
            k,
            v,
            logit_scale,
            bias,
            mask,
            o,
            lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            *bias_strides,
            *mask_strides,
            lse.stride(0),
            lse.stride(1),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            n_windows,
            seqlen_q,
            seqlen_k,
            seqlen_q_rounded,
            d,
            seqlen_q // 32,
            seqlen_k // 32,
            scale_type,
            has_bias,
            has_mask,
            BLOCK_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=1,
            )
        return o, lse, logit_scale