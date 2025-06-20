import os
import sys
import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_win_attn_v2 import flash_win_attn_v2_func, win_attention_torch, win_attention_torch_compile


def test_attention_correctness(
    batch: int,
    nwindows: int,
    nheads: int,
    seqlen: int,
    d: int,
    use_logit_scale: bool = True,
    dtype: type = torch.float16,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize input tensors
    q, k, v = [
        torch.randn((batch * nwindows, nheads, seqlen, d), device=device, dtype=dtype, requires_grad=True)
        for _ in range(3)
    ]
    bias = torch.randn((nheads, seqlen, seqlen), device=device, dtype=dtype, requires_grad=True)
    mask = torch.zeros((nwindows, seqlen, seqlen), device=device).masked_fill(
        torch.randn((nwindows, seqlen, seqlen), device=device) < 0, -float("inf")
    ).to(device=device, dtype=dtype)
    if use_logit_scale:
        logit_scale = torch.randn((nheads, 1, 1), device=device, dtype=dtype, requires_grad=True)
    else:
        logit_scale = None

    # Compute outputs
    o_flash = flash_win_attn_v2_func(q, k, v, logit_scale, bias, mask)
    o_torch = win_attention_torch(q, k, v, logit_scale, bias, mask)
    o_compile = win_attention_torch_compile(q, k, v, logit_scale, bias, mask)

    # Compute max forward errors
    max_forward_error_flash = (o_flash - o_torch).abs().max().item()
    max_forward_error_compile = (o_compile - o_torch).abs().max().item()

    # Compute loss and backpropagate for flash attention
    loss_flash = (0 - o_flash.sum())
    loss_flash.backward()
    dq_flash, dk_flash, dv_flash, dbias_flash = [x.grad.clone() for x in [q, k, v, bias]]
    if use_logit_scale:
        dscale_flash = logit_scale.grad.clone()

    # Reset gradients
    for x in [q, k, v, bias]:
        x.grad = None
    if use_logit_scale:
        logit_scale.grad = None

    # Compute loss and backpropagate for torch attention
    loss_torch = (0 - o_torch.sum())
    loss_torch.backward()
    dq_torch, dk_torch, dv_torch, dbias_torch = [x.grad.clone() for x in [q, k, v, bias]]
    if use_logit_scale:
        dscale_torch = logit_scale.grad.clone()

    # Reset gradients
    for x in [q, k, v, bias]:
        x.grad = None
    if use_logit_scale:
        logit_scale.grad = None

    # Compute loss and backpropagate for torch compile attention
    loss_compile = (0 - o_compile.sum())
    loss_compile.backward()
    dq_compile, dk_compile, dv_compile, dbias_compile = [x.grad.clone() for x in [q, k, v, bias]]
    if use_logit_scale:
        dscale_compile = logit_scale.grad.clone()

    # Reset gradients
    for x in [q, k, v, bias]:
        x.grad = None
    if use_logit_scale:
        logit_scale.grad = None

    max_grad_errors_flash = {
        "q": (dq_flash - dq_torch).abs().max().item(),
        "k": (dk_flash - dk_torch).abs().max().item(),
        "v": (dv_flash - dv_torch).abs().max().item(),
        "bias": (dbias_flash - dbias_torch).abs().max().item(),
    }
    max_grad_errors_flash["scale"] = (dscale_flash - dscale_torch).abs().max().item() if use_logit_scale else 0

    max_grad_errors_compile = {
        "q": (dq_compile - dq_torch).abs().max().item(),
        "k": (dk_compile - dk_torch).abs().max().item(),
        "v": (dv_compile - dv_torch).abs().max().item(),
        "bias": (dbias_compile - dbias_torch).abs().max().item(),
    }
    max_grad_errors_compile["scale"] = (dscale_compile - dscale_torch).abs().max().item() if use_logit_scale else 0

    # Print max errors
    print(f"✅ Max Forward Error Flash vs Torch: {max_forward_error_flash:.6f}")
    print(f"✅ Max Forward Error Torch Compile vs Troch: {max_forward_error_compile:.6f}")

    print(f"✅ Max Gradient Errors Flash vs Torch:")
    for param, err in max_grad_errors_flash.items():
        print(f"   {param.upper()} : {err:.6f}")

    print(f"✅ Max Gradient Errors Torch Compile vs Torch:")
    for param, err in max_grad_errors_compile.items():
        print(f"   {param.upper()} : {err:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FWA-2 correctness with customizable parameters.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--nwindows", type=int, default=16, help="Number of windows")
    parser.add_argument("--nheads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--seqlen", type=int, default=64, help="Sequence length (window size squarred)")
    parser.add_argument("--d", type=int, default=32, help="Dimension of attention features")
    parser.add_argument("--no_logit_scale", action="store_false", dest="use_logit_scale",
                        help="Disable the use of logit scale")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"],
                        help="Data type for tensors (float16 or bfloat16)")

    args = parser.parse_args()

    # Map string dtype to torch dtype
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    selected_dtype = dtype_map[args.dtype]

    print("Running correctness tests...")
    print("-" * 50)
    test_attention_correctness(
        batch=args.batch,
        nwindows=args.nwindows,
        nheads=args.nheads,
        seqlen=args.seqlen,
        d=args.d,
        use_logit_scale=args.use_logit_scale,
        dtype=selected_dtype,
    )