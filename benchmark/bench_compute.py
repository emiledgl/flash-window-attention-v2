import torch
import triton

from utils import calculate_cosine_attention_flops
from config import benchmark_configs
from win_attention_func import flash_win_attn_v2, win_attention_torch, win_attention_torch_compile


def test_attention_correctness(use_logit_scale=True, dtype=torch.float16):

    #torch.manual_seed(0)
    device = torch.device("cuda")

    # Fixed sequence length
    batch, nwindows, nheads, seqlen, d = 8, 16, 6, 1024, 32

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
    o_flash = flash_win_attn_v2(q, k, v, logit_scale, bias, mask)
    o_torch = win_attention_torch(q, k, v, logit_scale, bias, mask)
    o_compile = win_attention_torch_compile(q, k, v, logit_scale, bias, mask)

    # Compute max forward errors
    max_forward_error_torch = (o_flash - o_torch).abs().max().item()
    max_forward_error_compile = (o_flash - o_compile).abs().max().item()

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

    max_grad_errors_torch = {
        "q": (dq_flash - dq_torch).abs().max().item(),
        "k": (dk_flash - dk_torch).abs().max().item(),
        "v": (dv_flash - dv_torch).abs().max().item(),
        "bias": (dbias_flash - dbias_torch).abs().max().item(),
    }
    max_grad_errors_torch["scale"] = (dscale_flash - dscale_torch).abs().max().item() if use_logit_scale else 0

    max_grad_errors_compile = {
        "q": (dq_flash - dq_compile).abs().max().item(),
        "k": (dk_flash - dk_compile).abs().max().item(),
        "v": (dv_flash - dv_compile).abs().max().item(),
        "bias": (dbias_flash - dbias_compile).abs().max().item(),
    }
    max_grad_errors_compile["scale"] = (dscale_flash - dscale_compile).abs().max().item() if use_logit_scale else 0

    # Print max errors
    print(f"✅ Max Forward Error vs Torch: {max_forward_error_torch:.6f}")
    print(f"✅ Max Forward Error vs Torch Compile: {max_forward_error_compile:.6f}")

    print(f"✅ Max Gradient Errors vs Torch:")
    for param, err in max_grad_errors_torch.items():
        print(f"   {param.upper()} : {err:.6f}")

    print(f"✅ Max Gradient Errors vs Torch Compile:")
    for param, err in max_grad_errors_compile.items():
        print(f"   {param.upper()} : {err:.6f}")

    print(f"Max Scale Gradient Torch vs Torch Compile: {(dscale_torch - dscale_compile).abs().max().item():.6f}")


@triton.testing.perf_report(benchmark_configs)
def bench_attention_comparison(BATCH, W, H, seqlen, HEAD_DIM, mode, metric, provider, dtype=torch.float16, use_logit_scale=True):

    device = torch.device("cuda")

    # Initialize tensors with same shape ordering as test_attention_correctness
    q = torch.randn((BATCH * W, H, seqlen, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH * W, H, seqlen, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH * W, H, seqlen, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn((H, seqlen, seqlen), device=device, dtype=dtype, requires_grad=True)
    mask = torch.zeros((W, seqlen, seqlen), device=device).masked_fill(
        torch.randn((W, seqlen, seqlen), device=device) < 0, -float("inf")
    ).to(device=device, dtype=dtype)
    if use_logit_scale:
        logit_scale = torch.randn((H, 1, 1), device=device, dtype=dtype, requires_grad=True)
    else:
        logit_scale = None

    def forward_backward_triton():
        out = flash_win_attn_v2(q, k, v, logit_scale, bias, mask)
        loss = (0 - out.sum())
        loss.backward()
        torch.cuda.synchronize()
        # Reset gradients for next iteration
        q.grad = None
        k.grad = None
        v.grad = None
        if use_logit_scale:
            logit_scale.grad = None
        bias.grad = None

    def forward_backward_torch():
        out = win_attention_torch(q, k, v, logit_scale, bias, mask)
        loss = (0 - out.sum())
        loss.backward()
        torch.cuda.synchronize()
        # Reset gradients for next iteration
        q.grad = None
        k.grad = None
        v.grad = None
        if use_logit_scale:
            logit_scale.grad = None
        bias.grad = None

    def forward_backward_torch_compile():
        out = win_attention_torch_compile(q, k, v, logit_scale, bias, mask)
        loss = (0 - out.sum())
        loss.backward()
        torch.cuda.synchronize()
        # Reset gradients for next iteration
        q.grad = None
        k.grad = None
        v.grad = None
        if use_logit_scale:
            logit_scale.grad = None
        bias.grad = None

    if mode == "fwd":
        if provider == "triton-win-attn-2":
            fn = lambda: flash_win_attn_v2(q, k, v, logit_scale, bias, mask)
        elif provider == "torch-win-attn-2":
            fn = lambda: win_attention_torch(q, k, v, logit_scale, bias, mask)
        elif provider == "torch-compile-win-attn-2":
            fn = lambda: win_attention_torch_compile(q, k, v, logit_scale, bias, mask)
        else:
            raise Exception("Invalid provider \"{}\"".format(provider))
    elif mode == "bwd":
        if provider == "triton-win-attn-2":
            fn = forward_backward_triton
        elif provider == "torch-win-attn-2":
            fn = forward_backward_torch
        elif provider == "torch-compile-win-attn-2":
            fn = forward_backward_torch_compile
        else:
            raise Exception("Invalid provider \"{}\"".format(provider))
    else:
        raise Exception("Invalid mode \"{}\"".format(mode))

    # Run benchmark
    ms = triton.testing.do_bench(fn)

    if metric == "TFLOPS":
        # Calculate FLOPS
        total_flops_fwd = calculate_cosine_attention_flops(BATCH, W, H, seqlen, HEAD_DIM)

        # Backward pass has roughly 2x the FLOPS of forward pass
        total_flops = total_flops_fwd * (3 if mode == "bwd" else 1)

        # Calculate TFLOPS
        tflops = total_flops * 1e-12 / (ms * 1e-3)

        # Return TFLOPS for plotting
        return tflops
    else:
        return ms


if __name__ == "__main__":
    print("Running correctness tests...")
    print("-" * 50)
    test_attention_correctness()

    print("\nRunning performance benchmarks...")
    print("-" * 50)
    df = bench_attention_comparison.run(save_path="benchmark/results", print_data=True, return_df=True)