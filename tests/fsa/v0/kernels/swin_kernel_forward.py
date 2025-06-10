import triton
import triton.language as tl

# Q,K,V: (batch, heads, n_windows, seq, head_dim)
# B: (heads, seq, seq)
# M: (n_windows, seq, seq)
# -> O: (batch, heads, n_windows, seq, head_dim)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    bias,
    mask,
    O,
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

    # head_chunk = head_dim / chunk_dim
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

    # save output
    V_ptr = tl.make_block_ptr(
        base=V + block_start,
        shape=(seq, head_dim),
        strides=(head_dim, 1),
        offsets=(0, 0),
        block_shape=(seq_pad, chunk_dim),
        order=(1, 0),
    )
    index = block_start + tl.arange(0, seq_pad)[:, None] * head_dim + tl.arange(0, chunk_dim)[None, :]
    O_ptr = O + index
    for _ in range(head_chunk):
        v_data = tl.load(V_ptr, boundary_check=(0, 1), padding_option="zero")
        o_data = tl.dot(attn, v_data)
        o_data = o_data.cast(Q.dtype.element_ty)
        tl.store(O_ptr, o_data, mask=pad_mask[:, None])
        V_ptr = tl.advance(V_ptr, (0, chunk_dim))
        O_ptr += chunk_dim