import triton
import triton.language as tl

# Q,K,V: (batch, heads, n_windows, seq, head_dim)
# B: (heads, seq, seq)
# M: (n_windows, seq, seq)
# -> O: (batch, heads, n_windows, seq, head_dim)
@triton.jit
def _bwd_kernel(
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
    scale_qk: tl.constexpr,
    n_windows: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    head_chunk: tl.constexpr,
    chunk_dim: tl.constexpr,
    seq: tl.constexpr,
    seq_pad: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    window_id = tl.program_id(2)

    stride_window = seq * head_dim
    stride_head = stride_window * n_windows
    stride_batch = stride_head * heads
    
    block_start = batch_id * stride_batch + head_id * stride_head + window_id * stride_window
    pad_mask = tl.arange(0, seq_pad) < seq

    # load attn_mask
    # (n_windows, seq, seq)
    if mask is not None:
        Mask_ptr = tl.make_block_ptr(
            base=mask + window_id * seq * seq,
            shape=(seq, seq),
            strides=(seq, 1),
            offsets=(0, 0),
            block_shape=(seq_pad, seq_pad),
            order=(1, 0),
        )
        attn_mask = tl.load(Mask_ptr, boundary_check=(0, 1), padding_option="zero")
    
    # load bias
    # (heads, seq, seq)
    Bias_ptr = tl.make_block_ptr(
        base=bias + head_id * seq * seq,
        shape=(seq, seq),
        strides=(seq, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, seq_pad),
        order=(1, 0),
    )

    bias_data = tl.load(Bias_ptr, boundary_check=(0, 1), padding_option="zero")

    # compute attn matrix
    attn = tl.zeros((seq_pad, seq_pad), dtype=tl.float32)
    Q_ptr = tl.make_block_ptr(
        base=Q + block_start,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    K_ptr = tl.make_block_ptr(
        base=K + block_start,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    for _ in range(head_chunk):
        # load data
        q_data = tl.load(Q_ptr, boundary_check=(0, 1), padding_option="zero")
        k_data = tl.load(K_ptr, boundary_check=(0, 1), padding_option="zero")
        # dot of bf16 -> fp32
        attn = tl.dot(q_data, k_data.trans(1, 0), attn)
        Q_ptr = tl.advance(Q_ptr, (0, chunk_dim))
        K_ptr = tl.advance(K_ptr, (0, chunk_dim))

    attn *= scale_qk
    attn += bias_data

    if mask is not None:
        attn += attn_mask

    attn += tl.where(pad_mask[None, :], 0, -float("inf"))

    # softmax
    row_max = tl.max(attn, axis=1, keep_dims=True)
    attn = attn - row_max
    attn = tl.math.exp(attn)
    row_sum = tl.sum(attn, axis=1, keep_dims=True)
    attn = attn / row_sum
    attn = attn.cast(Q.dtype.element_ty)

    # compute d_V, d_attn
    d_attn = tl.zeros((seq_pad, seq_pad), dtype=tl.float32)
    d_O_ptr = tl.make_block_ptr(
        base=d_O + block_start,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    V_ptr = tl.make_block_ptr(
        base=V + block_start,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )

    index = block_start + tl.arange(0, seq_pad)[:, None] * head_dim + tl.arange(0, chunk_dim)[None, :]
    d_V_ptr = d_V + index

    for _ in range(head_chunk):
        # load data
        d_o_data = tl.load(d_O_ptr, boundary_check=(0, 1), padding_option="zero")
        v_data = tl.load(V_ptr, boundary_check=(0, 1), padding_option="zero")

        # accumulate d_attn
        d_attn = tl.dot(d_o_data, v_data.trans(1, 0), d_attn)
        d_v_data = tl.dot(attn.trans(1, 0), d_o_data).cast(Q.dtype.element_ty)
        tl.store(d_V_ptr, d_v_data, mask=pad_mask[:, None])

        d_O_ptr = tl.advance(d_O_ptr, (0, chunk_dim))
        V_ptr = tl.advance(V_ptr, (0, chunk_dim))
        d_V_ptr += chunk_dim

    attn_times_d_attn = attn.cast(tl.float32) * d_attn.cast(tl.float32)
    attn_sum = tl.sum(attn_times_d_attn, axis=1, keep_dims=True)
    d_attn = attn.cast(tl.float32) * (d_attn.cast(tl.float32) - attn_sum)
    d_attn = d_attn.cast(Q.dtype.element_ty)

    # compute d_bias
    # (heads, seq, seq)
    index_bias = head_id * seq * seq + tl.arange(0, seq_pad)[:, None] * seq + tl.arange(0, seq_pad)[None, :]
    d_Bias_ptr = d_bias + index_bias
    tl.atomic_add(d_Bias_ptr, d_attn.cast(bias.dtype.element_ty), mask=(pad_mask[:, None] & pad_mask[None, :]))

    # compute d_Q, d_K
    Q_ptr = tl.make_block_ptr(
        base=Q + block_start,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    K_ptr = tl.make_block_ptr(
        base=K + block_start,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    d_Q_ptr = d_Q + index
    d_K_ptr = d_K + index

    for _ in range(head_chunk):
        # load data
        q_data = tl.load(Q_ptr, boundary_check=(0, 1), padding_option="zero")
        k_data = tl.load(K_ptr, boundary_check=(0, 1), padding_option="zero")

        d_q_data = tl.dot(d_attn.cast(tl.float32), k_data.cast(tl.float32)) * scale_qk
        d_q_data = d_q_data.cast(q_data.dtype)
        tl.store(d_Q_ptr, d_q_data, mask=pad_mask[:, None])

        d_k_data = tl.dot(d_attn.cast(tl.float32).trans(1, 0), q_data.cast(tl.float32)) * scale_qk
        d_k_data = d_k_data.cast(k_data.dtype)
        tl.store(d_K_ptr, d_k_data, mask=pad_mask[:, None])

        Q_ptr = tl.advance(Q_ptr, (0, chunk_dim))
        K_ptr = tl.advance(K_ptr, (0, chunk_dim))
        d_Q_ptr += chunk_dim
        d_K_ptr += chunk_dim