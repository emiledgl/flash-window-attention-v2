import torch
import triton

from .kernels import _fwd_kernel, _bwd_preprocess, _bwd_kernel

class FlashSwinAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, bias, mask, sm_scale):

        BATCH_SIZE, NUM_HEADS, NUM_WINDOWS, SEQ_LEN, HEAD_DIM = Q.shape
        O = torch.empty_like(Q)

        BLOCK_M = 64
        BLOCK_N = 64

        grid = (
            triton.cdiv(SEQ_LEN, BLOCK_M),
            BATCH_SIZE * NUM_HEADS * NUM_WINDOWS,
            1,
        )
        L = torch.empty((BATCH_SIZE * NUM_HEADS * NUM_WINDOWS, SEQ_LEN), device=Q.device, dtype=torch.float32)

        num_warps = 4 if HEAD_DIM <= 64 else 8
        _fwd_kernel[grid](
            Q,
            K,
            V,
            bias,
            mask,
            sm_scale,
            L,
            O,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            Q.stride(4),
            NUM_HEADS=NUM_HEADS,
            NUM_WINDOWS=NUM_WINDOWS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N, 
            num_warps=num_warps,
            num_stages=2,
            )

        ctx.save_for_backward(Q, K, V, bias, mask, O, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        return O

    @staticmethod
    def backward(ctx, dO):
        
        Q, K, V, bias, mask, O, L = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        BATCH_SIZE, NUM_HEADS, NUM_WINDOWS, SEQ_LEN, HEAD_DIM = Q.shape
        BLOCK_M = 128
        BLOCK_N = 128

        num_blocks_n = triton.cdiv(SEQ_LEN, BLOCK_N)

        # make contigious
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        dO = dO.contiguous()

        dQ = torch.zeros((num_blocks_n,) + Q.shape, device=Q.device, dtype=Q.dtype)
        stride_dq_all = dQ.stride(0)

        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        dbias_full = torch.zeros((BATCH_SIZE, NUM_HEADS, NUM_WINDOWS, SEQ_LEN, SEQ_LEN), device=Q.device, dtype=Q.dtype)
        delta = torch.empty_like(L)

        if dbias_full is not None:
            stride_dbias_batch = dbias_full.stride(0)
            stride_dbias_heads = dbias_full.stride(1)
            stride_dbias_win = dbias_full.stride(2)
        else:
            stride_dbias_batch = 0
            stride_dbias_heads = 0
            stride_dbias_win = 0

        _bwd_preprocess[(ctx.grid[0], ctx.grid[1], )](
            O,
            dO,
            delta,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            Q.stride(4),
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM,
            BLOCK_M=BLOCK_M,
        )

        _bwd_kernel[(ctx.grid[1], num_blocks_n, )](
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
            dbias_full,
            L,
            delta,
            stride_dq_all,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            Q.stride(3),
            Q.stride(4),
            stride_dbias_batch,
            stride_dbias_heads,
            stride_dbias_win,
            num_blocks_m=ctx.grid[0],
            NUM_HEADS=NUM_HEADS,
            NUM_WINDOWS=NUM_WINDOWS,
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM, 
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=1,
        )

        dQ = dQ.sum(dim=0)

        # Sum the dbias over batch and window dimensions to get the final gradient
        dbias = dbias_full.sum(dim=[0, 2])

        return dQ, dK, dV, dbias, None, None


flash_swin_attn = FlashSwinAttention.apply