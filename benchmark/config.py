import triton

# Benchmarking configuration
BATCH, N_WINDOWS, N_HEADS, HEAD_DIM = 8, 4, 6, 32  # Aligned with test_attention_correctness

benchmark_configs = [
    triton.testing.Benchmark(
        x_names=["seqlen"],
        x_vals=[i**2 for i in [8, 12, 16, 24, 32, 40]],
        line_arg="provider",
        line_vals=[
            "triton-win-attn-2",
            "torch-win-attn-2",
            "torch-compile-win-attn-2",
        ],
        line_names=[
            "Triton-Win-Attn-2 (TFLOPS)",
            "PyTorch-Win-Attn-2 (TFLOPS)",
            "PyTorch-Compile-Win-Attn-2 (TFLOPS)",
        ],
        styles=[
            ("red", "-"),
            ("blue", "--"),
            ("cyan", "-."),
        ],
        ylabel="TFLOPS",  # Changed from Time (ms) to TFLOPS
        plot_name=f"win-attention-comparison-batch{BATCH}-window{N_WINDOWS}-head{N_HEADS}-d{HEAD_DIM}-{mode}",
        args={
            "BATCH": BATCH,
            "W": N_WINDOWS,
            "H": N_HEADS,
            "HEAD_DIM": HEAD_DIM,
            "mode": mode,
        }
    )
    for mode in ["fwd", "bwd"]
]