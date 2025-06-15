import os
import sys
import torch
import triton

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import calculate_cosine_attention_flops
from config import benchmark_configs

from flash_win_attn_v2 import flash_win_attn_v2_func, win_attention_torch, win_attention_torch_compile


@triton.testing.perf_report(benchmark_configs)
def bench_attention_comparison(BATCH, W, H, seqlen, HEAD_DIM, mode, provider, dtype=torch.float16, use_logit_scale=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        out = flash_win_attn_v2_func(q, k, v, logit_scale, bias, mask)
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
            fn = lambda: flash_win_attn_v2_func(q, k, v, logit_scale, bias, mask)
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

    # Calculate FLOPS
    total_flops_fwd = calculate_cosine_attention_flops(BATCH, W, H, seqlen, HEAD_DIM)

    # Backward pass has roughly 2x the FLOPS of forward pass
    total_flops = total_flops_fwd * (3 if mode == "bwd" else 1)

    # Calculate TFLOPS
    tflops = total_flops * 1e-12 / (ms * 1e-3)

    # Return TFLOPS for plotting
    return tflops


if __name__ == "__main__":

    print("\nRunning performance benchmarks...")
    print("-" * 50)
    df = bench_attention_comparison.run(save_path="benchmark/results", print_data=True, return_df=True)