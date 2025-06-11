import math
import torch
from .kernels import _flash_attn_forward, _flash_attn_backward

class FlashWindowAttentionV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, logit_scale=None, bias=None, mask=None):

        # logit scale assertions
        if logit_scale is not None:
            assert isinstance(logit_scale, torch.Tensor), "Make sure logit_scale is a tensor if defined in the paramaters."
            scale_type = "tensor"
            logit_scale = logit_scale if logit_scale.stride(-1) == 1 else logit_scale.contiguous()
        else:
            scale_type = "scalar"
            logit_scale = 1.0 / math.sqrt(q.shape[-1])
            logit_scale = torch.tensor(logit_scale, device=q.device)

        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]

        o, lse, logit_scale = _flash_attn_forward(
            q, k, v, logit_scale, bias=bias, mask=mask, scale_type=scale_type,
        )

        ctx.save_for_backward(q, k, v, o, lse, logit_scale, bias, mask)
        ctx.scale_type = scale_type
        return o

    @staticmethod
    def backward(ctx, do):

        q, k, v, o, lse, logit_scale, bias, mask = ctx.saved_tensors
        scale_type = ctx.scale_type
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dscale = torch.empty_like(logit_scale) if scale_type == "tensor" else None
        db = torch.empty_like(bias) if bias is not None else None
        _flash_attn_backward(
            do,
            q,
            k,
            v,
            logit_scale,
            o,
            lse,
            dq,
            dk,
            dv,
            dscale,
            db,
            bias=bias,
            mask=mask,
            scale_type=scale_type
        )
        return dq, dk, dv, dscale, db, None
    

flash_win_attn_v2 = FlashWindowAttentionV2.apply