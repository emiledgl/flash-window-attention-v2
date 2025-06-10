import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    bias,
    mask,
    sm_scale,
    L,
    O,
    stride_batch,
    stride_heads,
    stride_win,
    stride_seq,
    stride_hdim,
    NUM_HEADS: tl.constexpr,
    NUM_WINDOWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    bhw_id = tl.program_id(1)
    bh_id = bhw_id // NUM_WINDOWS
    batch_id = bh_id // NUM_HEADS
    head_id = bh_id % NUM_HEADS
    window_id = bhw_id % NUM_WINDOWS
    qvk_offset = batch_id * stride_batch + head_id * stride_heads + window_id * stride_win
    log2e: tl.constexpr = 1.4426950408889634  # log2(e)
    
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_seq, stride_hdim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_hdim, stride_seq),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_seq, stride_hdim),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0)
    )
    
    # Initialize offsets with mask for valid positions
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < SEQ_LEN  # Mask for valid positions in the query sequence
    # Initialize accumulators and tracking variables
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Scale for softmax
    qk_scale = sm_scale * log2e
    
    # Load query with masking
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q = (q * qk_scale).to(tl.float16)
    
    # Loop over k, v and update accumulator
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # Calculate mask for valid positions in this key/value block
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < SEQ_LEN  # Mask for valid positions in key/value sequence
        pad_mask = mask_m[:, None] & mask_n[None, :]

        # Load k, v
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Compute qk with attention masking
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        # Apply bias
        if bias is not None:
            bias_ptrs = bias + head_id * SEQ_LEN * SEQ_LEN + offs_m[:, None] * SEQ_LEN + offs_n[None, :]
            attn_bias = tl.load(bias_ptrs, mask=pad_mask, other=0.0)
            attn_bias *= log2e
            qk += attn_bias
        
        # Apply shifted window masking
        if mask is not None:
            mask_ptrs = mask + window_id * SEQ_LEN * SEQ_LEN + offs_m[:, None] * SEQ_LEN + offs_n[None, :]
            attn_mask = tl.load(mask_ptrs, mask=pad_mask, other=float("-inf"))
            qk += attn_mask
        # Apply sequence length masking (ensure we don't attend to padding)
        else:
            qk = tl.where(pad_mask, qk, float("-inf"))
        
        # Compute scaling constant
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        
        # Scale and update accumulator
        acc_scale = l_i * 0 + alpha  # Workaround for compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        
        # Update m_i and l_i
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        
        # Update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    # Final normalization
    acc = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
    
    # Write back l and m with masking
    l_ptrs = L + bhw_id * SEQ_LEN + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=mask_m)
    
    offs_m_2d = offs_m[:, None]
    offs_h = tl.arange(0, HEAD_DIM)[None, :]
    
    # Calculate output pointer for each element manually
    o_ptrs = O + qvk_offset + offs_m_2d * stride_seq + offs_h * stride_hdim
    tl.store(o_ptrs, acc.to(tl.float16), mask=mask_m[:, None])