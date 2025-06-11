import triton

# Benchmarking configuration
BATCH, N_WINDOWS, N_HEADS, HEAD_DIM = 16, 16, 6, 32  # Aligned with test_attention_correctness

benchmark_configs = []

for mode in ["fwd", "bwd"]:
    for metric in ["TFLOPS", "TIME-ms"]:
        metric_suffix = metric.split("-")[-1]
        metric_prefix = metric.split("-")[0]
        
        config = triton.testing.Benchmark(
            x_names=["seqlen"],
            x_vals=[i**2 for i in [8, 12, 16, 24]],
            line_arg="provider",
            line_vals=[
                "triton-win-attn-2",
                "torch-win-attn-2",
                "torch-compile-win-attn-2",
            ],
            line_names=[
                f"Triton-Win-Attn-2 ({metric_suffix})",
                f"PyTorch-Win-Attn-2 ({metric_suffix})",
                f"PyTorch-Compile-Win-Attn-2 ({metric_suffix})",
            ],
            styles=[
                ("red", "-"),
                ("blue", "--"),
                ("cyan", "-."),
            ],
            ylabel=f"{metric_prefix}",
            plot_name=f"win-attention-{metric_prefix.lower()}-comparison-batch{BATCH}-window{N_WINDOWS}-head{N_HEADS}-d{HEAD_DIM}-{mode}",
            args={
                "BATCH": BATCH,
                "W": N_WINDOWS,
                "H": N_HEADS,
                "HEAD_DIM": HEAD_DIM,
                "mode": mode,
                "metric": metric,
            }
        )
        benchmark_configs.append(config)