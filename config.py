import triton

# Benchmarking configuration
BATCH, N_WINDOWS, N_HEADS, HEAD_DIM = 4, 4, 8, 32  # Aligned with test_attention_correctness

benchmark_configs = [
    triton.testing.Benchmark(
        x_names=["seqlen"],
        x_vals=[i**2 for i in [8, 12, 16, 24, 32]],
        line_arg="provider",
        line_vals=[
            "triton-cos-attn",
            "torch-cos-attn",
            "torch-compile-cos-attn",
        ],
        line_names=[
            "Triton-Cos-Attn (FLOPS)",
            "PyTorch-Cos-Attn (FLOPS)",
            "PyTorch-Compile-Cos-Attn (FLOPS)",
        ],
        styles=[
            ("red", "-"),
            ("blue", "--"),
            ("cyan", "-."),
        ],
        ylabel="TFLOPS",  # Changed from Time (ms) to TFLOPS
        plot_name=f"cos-attention-comparison-batch{BATCH}-window{N_WINDOWS}-head{N_HEADS}-d{HEAD_DIM}-{mode}",
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