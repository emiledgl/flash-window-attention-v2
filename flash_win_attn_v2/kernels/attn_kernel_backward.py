import math
import torch
import triton
import triton.language as tl

@triton.jit
def _bwd_preprocess_do_o_dot(
        O,
        DO,
        Delta,
        stride_obw,
        stride_oh,
        stride_om,
        stride_dobw,
        stride_doh,
        stride_dom,
        stride_dbw,
        stride_dh,
        seqlen_q,
        headdim,
        BLOCK_M: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bw = tl.program_id(1)
    off_h = tl.program_id(2)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(
        O + off_bw * stride_obw + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_bw * stride_dobw
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(
        Delta + off_bw * stride_dbw + off_h * stride_dh + offs_m, 
        delta,
        mask=offs_m < seqlen_q
    )


@triton.jit
def _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
):
    # Store dk and dv
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))

@triton.jit
def _bwd_kernel_one_col_block(
        start_n,
        Q,
        K,
        V,
        Scale,
        Bias,
        Mask,
        DO,
        DQ,
        DK,
        DV,
        DB,
        LSE,
        D,
        stride_qm,
        stride_kn,
        stride_vn,
        stride_bm,
        stride_mm,
        stride_dbm,
        stride_dom,
        stride_dqm,
        stride_dkn,
        stride_dvn,
        seqlen_q,
        seqlen_k,
        headdim,
        ATOMIC_ADD: tl.constexpr,
        SCALE_TYPE: tl.constexpr,  # "scalar" or "tensor"
        HAS_BIAS: tl.constexpr,
        HAS_MASK: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
) -> float:
    # initialize row/col offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    MAX_LOG: tl.constexpr = tl.log(100.0)

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_d[None, :])

    if HAS_BIAS:
        bias_ptrs = Bias + (offs_m[:, None] * stride_bm + offs_n[None, :])
        dbias_ptrs = DB + (offs_m[:, None] * stride_dbm + offs_n[None, :])
    if HAS_MASK:
        mask_ptrs = Mask + (offs_m[:, None] * stride_mm + offs_n[None, :])

    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk_n = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    
    # Load K and V once for this column block
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
    
    # normalize k
    norm_k = tl.sqrt(tl.sum(k.to(tl.float32) * k.to(tl.float32), axis=1, keep_dims=True) + 1e-12)
    k_n = k.to(tl.float32) / norm_k

    # Scale for softmax
    log_scale = tl.load(Scale).to(tl.float32)
    if SCALE_TYPE == "tensor":
        logit_scale = tl.exp(tl.minimum(log_scale, MAX_LOG))
    else:
        logit_scale = log_scale
    qk_scale = logit_scale

    # initialize dscale
    dscale = tl.zeros([], dtype=tl.float32) if SCALE_TYPE == "tensor" else None

    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(0, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
            
        # load q, do on-chip
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
            do = tl.load(do_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
                do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
                do = tl.load(
                    do_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        # normalize q
        norm_q = tl.sqrt(tl.sum(q.to(tl.float32) * q.to(tl.float32), axis=1, keep_dims=True) + 1e-12)
        q_n = q.to(tl.float32) / norm_q
        
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q_n.to(q.dtype), tl.trans(k_n.to(k.dtype)))
        qk_scaled = qk * qk_scale

        if HAS_BIAS:
            if EVEN_M & EVEN_N:
                bias = tl.load(bias_ptrs).to(tl.float32)
            else:
                bias = tl.load(
                    bias_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                    other=0.0,
                ).to(tl.float32)
            #bias *= LOG2E
            qk_scaled += bias
            
        if HAS_MASK:
            if EVEN_M & EVEN_N:
                mask = tl.load(mask_ptrs).to(tl.float32)
            else:
                mask = tl.load(
                    mask_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                    other=float("-inf"),
                ).to(tl.float32)
            qk_scaled += mask
        else:
            if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
                qk_scaled = tl.where(offs_n[None, :] < seqlen_k, qk_scaled, float("-inf"))
        
        # Apply causal mask if needed
        if not EVEN_M:
            qk_scaled = tl.where(offs_m_curr[:, None] < seqlen_q, qk_scaled, float("-inf"))
        
        # Compute softmax
        lse_i = tl.load(LSE + offs_m_curr, mask=offs_m_curr < seqlen_q, other=0.0)
        p = tl.exp(qk_scaled - lse_i[:, None])

        # compute dv
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        
        # compute dp = dot(v, do)
        dp = tl.dot(do, tl.trans(v))
        
        # compute ds = p * (dp - delta[:, None]) * logit_scale
        Di = tl.load(D + offs_m_curr, mask=offs_m_curr < seqlen_q, other=0.0)
        db = p * (dp - Di[:, None])
        ds = db * logit_scale

        # compute dscale = sum(ds * qk)
        if SCALE_TYPE == "tensor":
            dscale += tl.sum(ds * qk)
        
        # compute dk = dot(ds.T, q)
        dk_n += tl.dot(tl.trans(ds.to(q.dtype)), q_n.to(q.dtype))
        
        # compute dq
        dq_n = tl.dot(ds.to(q.dtype), k_n.to(k.dtype))
        dq_contrib = dq_n / norm_q - (q * tl.sum(dq_n * q_n, axis=1, keep_dims=True)) / (norm_q * norm_q)

        if HAS_BIAS:
            if EVEN_M & EVEN_N:
                tl.atomic_add(dbias_ptrs, db)
            else:
                db_mask = (offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k)
                tl.atomic_add(dbias_ptrs, db, mask=db_mask)
        
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += dq_contrib
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += dq_contrib
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += dq_contrib
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
        else:
            if EVEN_M & EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq_contrib)
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq_contrib, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq_contrib,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    )
        
        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if HAS_BIAS:
            bias_ptrs += BLOCK_M * stride_bm
            dbias_ptrs += BLOCK_M * stride_dbm
        if HAS_MASK:
            mask_ptrs += BLOCK_M * stride_mm

    # write-back
    dk = dk_n / norm_k - (k * tl.sum(dk_n * k_n, axis=1, keep_dims=True)) / (norm_k * norm_k)
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
    )

    return dscale

# error in bias and scale gradient computation when using multiple autotune configs
@triton.autotune(
    configs=[
        # triton.Config(
        #     {"BLOCK_M": 64, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True},
        #     num_warps=4,
        #     num_stages=1,
        # ),
        # triton.Config(
        #     {"BLOCK_M": 64, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False},
        #     num_warps=4,
        #     num_stages=1,
        # ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True},
            num_warps=4,
            num_stages=1,
        ),
        # triton.Config(
        #     {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False},
        #     num_warps=4,
        #     num_stages=1,
        # ),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
        Q,
        K,
        V,
        Scale,
        Bias,
        Mask,
        DO,
        DQ,
        DK,
        DV,
        DScale,
        DB,
        LSE,
        D,
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
        stride_dbh, stride_dbm,  # bias gradient strides
        stride_dobw,
        stride_doh,
        stride_dom,
        stride_dqbw,
        stride_dqh,
        stride_dqm,
        stride_dkbw,
        stride_dkh,
        stride_dkn,
        stride_dvbw,
        stride_dvh,
        stride_dvn,
        stride_lse_bw,
        stride_lse_h,
        n_windows,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        headdim,
        CACHE_KEY_SEQLEN_Q,
        CACHE_KEY_SEQLEN_K,
        SCALE_TYPE: tl.constexpr,  # "scalar" or "tensor"
        HAS_BIAS: tl.constexpr,
        HAS_MASK: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        SEQUENCE_PARALLEL: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    off_bw = tl.program_id(1)
    off_h = tl.program_id(2)
    # offset pointers for batch/head
    Q += off_bw * stride_qbw + off_h * stride_qh
    K += off_bw * stride_kbw + off_h * stride_kh
    V += off_bw * stride_vbw + off_h * stride_vh
    DO += off_bw * stride_dobw + off_h * stride_doh
    DQ += off_bw * stride_dqbw + off_h * stride_dqh
    DK += off_bw * stride_dkbw + off_h * stride_dkh
    DV += off_bw * stride_dvbw + off_h * stride_dvh
    if Bias is not None:
        Bias += off_h * stride_bh
        DB += off_h * stride_dbh
    if Mask is not None:
        off_w = off_bw % n_windows
        Mask += off_w * stride_mw
    if SCALE_TYPE == "tensor":
        Scale += off_h
        DScale += off_h
        if not SEQUENCE_PARALLEL:
            dscale = tl.zeros([], dtype=tl.float32)
    # pointer to row-wise quantities in value-like data
    D += off_bw * stride_lse_bw + off_h * stride_lse_h
    LSE += off_bw * stride_lse_bw + off_h * stride_lse_h
    
    
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            dscale_contrib = _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                Scale,
                Bias,
                Mask,
                DO,
                DQ,
                DK,
                DV,
                DB,
                LSE,
                D,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_bm,
                stride_mm,
                stride_dbm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                SCALE_TYPE=SCALE_TYPE,
                HAS_BIAS=HAS_BIAS,
                HAS_MASK=HAS_MASK,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
            if SCALE_TYPE == "tensor":
                dscale += dscale_contrib
    else:
        start_n = tl.program_id(0)
        dscale = _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            Scale,
            Bias,
            Mask,
            DO,
            DQ,
            DK,
            DV,
            DB,
            LSE,
            D,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_bm,
            stride_mm,
            stride_dbm,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            seqlen_q,
            seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            SCALE_TYPE=SCALE_TYPE,
            HAS_BIAS=HAS_BIAS,
            HAS_MASK=HAS_MASK,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )  
    
    # store dscale
    if SCALE_TYPE == "tensor":
        tl.atomic_add(DScale, dscale.to(DScale.dtype.element_ty))


def _flash_attn_backward(
    do, q, k, v, logit_scale, o, lse, dq, dk, dv, dscale, db, bias=None, mask=None, scale_type="scalar",
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch_window_size, n_heads, seqlen_q, d = q.shape
    _, _, seqlen_k, _ = k.shape
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 64) * 64
    assert lse.shape == (batch_window_size, n_heads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    
    # Initialize dq with zeros instead of empty tensor
    dq_accum = torch.zeros_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK_M = 32 if seqlen_q <= 64 else 64
    BLOCK_N = 32 if seqlen_k <= 64 else 64
    # bug in backward pass when increasing nump_warps to 8
    num_warps = 4
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch_window_size, n_heads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        delta.stride(0),
        delta.stride(1),
        seqlen_q,
        d,
        BLOCK_M=64,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    if scale_type == "tensor":
        if dscale.stride(-1) != 1:
            dscale = dscale.contiguous()
        dscale_accum = torch.zeros_like(dscale, dtype=torch.float32)
    else:
        dscale_accum = None

    has_bias = bias is not None
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.shape == (n_heads, seqlen_q, seqlen_k)
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if db.stride(-1) != 1:
            db = db.contiguous()
        
    db_accum = torch.zeros_like(db, dtype=torch.float32) if has_bias else None
    bias_strides = (bias.stride(0), bias.stride(1)) if has_bias else (0, 0)
    dbias_strides = (db.stride(0), db.stride(1)) if has_bias else (0, 0)

    has_mask = mask is not None
    if has_mask:
        assert mask.dtype in [q.dtype, torch.float]
        assert mask.is_cuda
        if mask.stride(-1) != 1:
            mask = mask.contiguous()
    n_windows = mask.shape[0] if mask is not None else None
    mask_strides = (mask.stride(0), mask.stride(1)) if has_mask else (0, 0)

    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch_window_size,
        n_heads,
    )
    _bwd_kernel[grid](
        q,
        k,
        v,
        logit_scale,
        bias,
        mask,
        do,
        dq_accum,
        dk,
        dv,
        dscale_accum,
        db_accum,
        lse,
        delta,
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
        *dbias_strides,
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dq_accum.stride(0),
        dq_accum.stride(1),
        dq_accum.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        lse.stride(0),
        lse.stride(1),
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
        # SEQUENCE_PARALLEL=True,
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )

    dq.copy_(dq_accum.to(dq.dtype))
    if has_bias:
        db.copy_(db_accum.to(db.dtype))
    if scale_type == "tensor":
        clamp_mask = (
            logit_scale <= torch.log(torch.tensor(100.0, dtype=logit_scale.dtype, device=logit_scale.device))
        ).float()
        dscale_clamped = dscale_accum * clamp_mask
        dscale.copy_(dscale_clamped.to(dscale.dtype))