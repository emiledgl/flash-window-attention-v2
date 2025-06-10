import triton
import triton.language as tl

@triton.jit
def _bwd_preprocess(
    O,
    dO,
    Delta,
    stride_batch,
    stride_heads,
    stride_win,
    stride_seq,
    stride_hdim,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):

    start_m = tl.program_id(0)
    bhw_id = tl.program_id(1)
    qvk_offset = bhw_id * stride_win

    o_offset = O + qvk_offset
    do_offset = dO + qvk_offset
    d_offset = Delta + bhw_id * SEQ_LEN
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < SEQ_LEN
    offs_n = tl.arange(0, HEAD_DIM)

    # load
    o = tl.load(o_offset + offs_m[:, None] * stride_seq + offs_n[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    dO = tl.load(do_offset + offs_m[:, None] * stride_seq + offs_n[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    # compute
    delta = tl.sum(o * dO, axis=1)
    # write-back
    tl.store(d_offset + offs_m, delta, mask=mask_m)

@triton.jit
def _bwd_kernel_inner(
    Q,
    K,
    V,
    q_offset,
    k_offset,
    v_offset,
    bias_offset,
    mask_offset,
    sm_scale,
    do_offset,
    dq_offset,
    dk_offset,
    dv_offset,
    dbias_offset,
    l_offset,
    d_offset,
    stride_seq,
    stride_hdim,
    start_n,
    num_blocks_m,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
):
    # initialize row/col offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < SEQ_LEN
    off_hdim = tl.arange(0, HEAD_DIM)
    log2e: tl.constexpr = 1.4426950408889634  # log2(e)

    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # initialize pointers to value-like data
    k_ptrs = k_offset + (offs_n[:, None] * stride_seq + off_hdim[None, :] * stride_hdim)
    v_ptrs = v_offset + (offs_n[:, None] * stride_seq + off_hdim[None, :] * stride_hdim)
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    # Scale for softmax
    qk_scale = sm_scale * log2e

    # loop over rows
    for start_m in range(0, num_blocks_m * BLOCK_M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < SEQ_LEN
        pad_mask = mask_m[:, None] & mask_n[None, :]

        q_ptrs = q_offset + (offs_m[:, None] * stride_seq + off_hdim[None, :] * stride_hdim)
        do_ptrs = do_offset + (offs_m[:, None] * stride_seq + off_hdim[None, :] * stride_hdim)
        dq_ptrs = dq_offset + (offs_m[:, None] * stride_seq + off_hdim[None, :] * stride_hdim)

        # load q, k, v, do on-chip
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

        # recompute p = softmax(qk, dim=-1).T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= qk_scale

        # Apply bias
        if bias_offset is not None:
            bias_ptrs = bias_offset + offs_m[:, None] * SEQ_LEN + offs_n[None, :]
            attn_bias = tl.load(bias_ptrs, mask=pad_mask, other=0.0)
            attn_bias *= log2e
            qk += attn_bias
        
        # Apply shifted window masking
        if mask_offset is not None:
            mask_ptrs = mask_offset + offs_m[:, None] * SEQ_LEN + offs_n[None, :]
            attn_mask = tl.load(mask_ptrs, mask=pad_mask, other=float("-inf"))
            qk += attn_mask

        l_ptrs = l_offset + offs_m
        l_i = tl.load(l_ptrs, mask=mask_m)
        #l_i *= log2e
        p = tl.math.exp2(qk - l_i[:, None])

        # mask block in the cases where the data is smaller the block size
        qk = tl.where(pad_mask, qk, 0.0)

        # compute dv
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)

        # compute dp
        dp = tl.dot(do, tl.trans(v))

        # compute ds , ds = p * (dp - delta[:, None])
        d_ptrs = d_offset + offs_m
        Di = tl.load(d_ptrs, mask=mask_m)
        ds = p * (dp - Di[:, None])
        ds = tl.where(pad_mask, ds, 0.0).to(Q.dtype.element_ty)

        if bias_offset is not None:
            dbias_ptrs = dbias_offset + offs_m[:, None] * SEQ_LEN + offs_n[None, :]
            tl.store(dbias_ptrs, ds, mask=pad_mask)
            
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q) * sm_scale
        
        # compute dq
        dq = tl.dot(ds, k) * sm_scale
        tl.store(dq_ptrs, dq.to(Q.dtype.element_ty), mask=mask_m[:, None])

    # write-back dv and dk
    dk_ptrs = dk_offset + (offs_n[:, None] * stride_seq + off_hdim[None, :] * stride_hdim)
    dv_ptrs = dv_offset + (offs_n[:, None] * stride_seq + off_hdim[None, :] * stride_hdim)
    
    # write-back
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=mask_n[:, None])


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    bias,
    mask,
    sm_scale,
    dO,
    dQ,
    dK,
    dV,
    dBias,
    L,
    D,
    stride_dq_all,
    stride_batch,
    stride_heads,
    stride_win,
    stride_seq,
    stride_hdim,
    stride_dbias_batch,
    stride_dbias_heads, 
    stride_dbias_win,
    num_blocks_m,
    NUM_HEADS: tl.constexpr,
    NUM_WINDOWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
):  
    bhw_id = tl.program_id(0)
    start_n = tl.program_id(1)

    bh_id = bhw_id // NUM_WINDOWS
    batch_id = bh_id // NUM_HEADS
    head_id = bh_id % NUM_HEADS
    window_id = bhw_id % NUM_WINDOWS

    # offset pointers for batch/head
    q_offset = Q + batch_id * stride_batch + head_id * stride_heads + window_id * stride_win
    k_offset = K + batch_id * stride_batch + head_id * stride_heads + window_id * stride_win
    v_offset = V + batch_id * stride_batch + head_id * stride_heads + window_id * stride_win
    do_offset = dO + batch_id * stride_batch + head_id * stride_heads + window_id * stride_win

    if mask is not None:
        mask_offset = mask + window_id * SEQ_LEN * SEQ_LEN
    else:
        mask_offset = None

    if bias is not None:
        bias_offset = bias + head_id * SEQ_LEN * SEQ_LEN
        dbias_offset = dBias + batch_id * stride_dbias_batch + head_id * stride_dbias_heads + window_id * stride_dbias_win
    else:
        bias_offset, dbias_offset = None, None

    # pointer to row-wise quantities in value-like data
    d_offset = D + bhw_id * SEQ_LEN
    l_offset = L + bhw_id * SEQ_LEN

    dq_offset = dQ + start_n * stride_dq_all + batch_id * stride_batch + head_id * stride_heads + window_id * stride_win
    dk_offset = dK + batch_id * stride_batch + head_id * stride_heads + window_id * stride_win
    dv_offset = dV + batch_id * stride_batch + head_id * stride_heads + window_id * stride_win

    _bwd_kernel_inner(
        Q,
        K,
        V,
        q_offset,
        k_offset,
        v_offset,
        bias_offset,
        mask_offset,
        sm_scale,
        do_offset,
        dq_offset,
        dk_offset,
        dv_offset,
        dbias_offset,
        l_offset,
        d_offset,
        stride_seq,
        stride_hdim,
        start_n,
        num_blocks_m,
        SEQ_LEN,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
    )