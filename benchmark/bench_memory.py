import os
import torch
import csv
import numpy as np
import matplotlib.pyplot as plt

# Assuming these functions are defined in a separate file as in the original code
from win_attention_func import flash_win_attn_v2, win_attention_torch, win_attention_torch_compile

def benchmark_memory_allocation(
    seq_lens: list,
    batch_size: int,
    num_windows: int,
    num_heads: int,
    head_dim: int,
    num_trials: int = 10,
):
    """
    Benchmark memory usage of classic vs. flash attention implementations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("CUDA not available, skipping benchmark.")
        return {}
        
    dtype = torch.float16
    results = {}

    for seq_len in seq_lens:
        print(f"Benchmarking memory: batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}")

        # Create inputs
        q = torch.randn(batch_size * num_windows, num_heads, seq_len, head_dim,
                    dtype=dtype, device=device, requires_grad=True)
        k = torch.randn_like(q).requires_grad_(True)
        v = torch.randn_like(q).requires_grad_(True)
        bias = torch.randn(num_heads, seq_len, seq_len, device=device, requires_grad=True).half()
        mask = torch.zeros(num_windows, seq_len, seq_len, device=device)
        mask = mask.masked_fill(torch.rand(num_windows, seq_len, seq_len, dtype=dtype, device=device) < 0.5, float('-inf')).to(dtype=dtype, device=device)
        logit_scale = torch.randn((num_heads, 1, 1), dtype=dtype, device=device, requires_grad=True)

        # Detach and clone inputs for both attention types to ensure fair comparison
        q_classic, k_classic, v_classic, bias_classic = (t.detach().clone().requires_grad_(True) for t in (q, k, v, bias))
        logit_scale_classic = logit_scale.detach().clone().requires_grad_(True) if isinstance(logit_scale, torch.Tensor) else logit_scale

        q_compile, k_compile, v_compile, bias_compile = (t.detach().clone().requires_grad_(True) for t in (q, k, v, bias))
        logit_scale_compile = logit_scale.detach().clone().requires_grad_(True) if isinstance(logit_scale, torch.Tensor) else logit_scale

        q_flash, k_flash, v_flash, bias_flash = (t.detach().clone().requires_grad_(True) for t in (q, k, v, bias))
        logit_scale_flash = logit_scale.detach().clone().requires_grad_(True) if isinstance(logit_scale, torch.Tensor) else logit_scale

        # Warm up GPU
        for _ in range(3):
            _ = win_attention_torch(q_classic, k_classic, v_classic, logit_scale_classic, bias_classic, mask)
            _ = win_attention_torch_compile(q_compile, k_compile, v_compile, logit_scale_compile, bias_compile, mask)
            _ = flash_win_attn_v2(q_flash, k_flash, v_flash, logit_scale_flash, bias_flash, mask)

        # Benchmark forward pass memory
        classic_fwd_memory, compile_fwd_memory, flash_fwd_memory = [], [], []
        for _ in range(num_trials):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _ = win_attention_torch(q_classic, k_classic, v_classic, logit_scale_classic, bias_classic, mask)
            torch.cuda.synchronize()
            classic_fwd_memory.append(torch.cuda.max_memory_allocated() / (1024 ** 2))  # MB

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _ = win_attention_torch_compile(q_compile, k_compile, v_compile, logit_scale_compile, bias_compile, mask)
            torch.cuda.synchronize()
            compile_fwd_memory.append(torch.cuda.max_memory_allocated() / (1024 ** 2))  # MB

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _ = flash_win_attn_v2(q_flash, k_flash, v_flash, logit_scale_flash, bias_flash, mask)
            torch.cuda.synchronize()
            flash_fwd_memory.append(torch.cuda.max_memory_allocated() / (1024 ** 2))  # MB

        # Benchmark backward pass memory
        out1_for_grad = win_attention_torch(q_classic, k_classic, v_classic, logit_scale_classic, bias_classic, mask)
        grad_output = torch.randn_like(out1_for_grad)

        classic_bwd_memory, compile_bwd_memory, flash_bwd_memory = [], [], []
        for _ in range(num_trials):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            out1 = win_attention_torch(q_classic, k_classic, v_classic, logit_scale_classic, bias_classic, mask)
            out1.backward(grad_output) # Retain graph if needed for multiple backward passes
            torch.cuda.synchronize()
            classic_bwd_memory.append(torch.cuda.max_memory_allocated() / (1024 ** 2))  # MB

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            out2 = win_attention_torch_compile(q_compile, k_compile, v_compile, logit_scale_compile, bias_compile, mask)
            out2.backward(grad_output) # Retain graph if needed for multiple backward passes
            torch.cuda.synchronize()
            compile_bwd_memory.append(torch.cuda.max_memory_allocated() / (1024 ** 2))  # MB

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            out3 = flash_win_attn_v2(q_flash, k_flash, v_flash, logit_scale_flash, bias_flash, mask)
            out3.backward(grad_output)
            torch.cuda.synchronize()
            flash_bwd_memory.append(torch.cuda.max_memory_allocated() / (1024 ** 2))  # MB

        # Calculate mean and std for memory
        classic_fwd_mem_mean, classic_fwd_mem_std = np.mean(classic_fwd_memory), np.std(classic_fwd_memory)
        compile_fwd_mem_mean, compile_fwd_mem_std = np.mean(compile_fwd_memory), np.std(compile_fwd_memory)
        flash_fwd_mem_mean, flash_fwd_mem_std = np.mean(flash_fwd_memory), np.std(flash_fwd_memory)
        classic_bwd_mem_mean, classic_bwd_mem_std = np.mean(classic_bwd_memory), np.std(classic_bwd_memory)
        compile_bwd_mem_mean, compile_bwd_mem_std = np.mean(compile_bwd_memory), np.std(compile_bwd_memory)
        flash_bwd_mem_mean, flash_bwd_mem_std = np.mean(flash_bwd_memory), np.std(flash_bwd_memory)
        
        # Calculate memory reduction
        mem_reduc_classic_fwd = classic_fwd_mem_mean / flash_fwd_mem_mean if flash_fwd_mem_mean > 0 else float('inf')
        mem_reduc_compile_fwd = compile_fwd_mem_mean / flash_fwd_mem_mean if flash_fwd_mem_mean > 0 else float('inf')
        mem_reduc_classic_bwd = classic_bwd_mem_mean / flash_bwd_mem_mean if flash_bwd_mem_mean > 0 else float('inf')
        mem_reduc_compile_bwd = compile_bwd_mem_mean / flash_bwd_mem_mean if flash_bwd_mem_mean > 0 else float('inf')

        print(f"  Classic Forward Memory: {classic_fwd_mem_mean:.2f} ± {classic_fwd_mem_std:.2f} MB")
        print(f"  Compile Forward Memory: {compile_fwd_mem_mean:.2f} ± {compile_fwd_mem_std:.2f} MB")
        print(f"  Flash Forward Memory:   {flash_fwd_mem_mean:.2f} ± {flash_fwd_mem_std:.2f} MB")
        print(f"  Classic Backward Memory: {classic_bwd_mem_mean:.2f} ± {classic_bwd_mem_std:.2f} MB")
        print(f"  Compile Backward Memory: {compile_bwd_mem_mean:.2f} ± {compile_bwd_mem_std:.2f} MB")
        print(f"  Flash Backward Memory:   {flash_bwd_mem_mean:.2f} ± {flash_bwd_mem_std:.2f} MB")
        print(f"  Classic Forward Memory Reduction: {mem_reduc_classic_fwd:.2f}x")
        print(f"  Compile Forward Memory Reduction: {mem_reduc_compile_fwd:.2f}x")
        print(f"  Classic Backward Memory Reduction: {mem_reduc_classic_bwd:.2f}x")
        print(f"  Compile Backward Memory Reduction: {mem_reduc_compile_bwd:.2f}x")

        key = seq_len
        results[key] = {
            'classic_fwd_mem': (classic_fwd_mem_mean, classic_fwd_mem_std),
            'compile_fwd_mem': (compile_fwd_mem_mean, compile_fwd_mem_std),
            'flash_fwd_mem': (flash_fwd_mem_mean, flash_fwd_mem_std),
            'classic_bwd_mem': (classic_bwd_mem_mean, classic_bwd_mem_std),
            'compile_bwd_mem': (compile_bwd_mem_mean, compile_bwd_mem_std),
            'flash_bwd_mem': (flash_bwd_mem_mean, flash_bwd_mem_std),
            'mem_reduc_classic_fwd': mem_reduc_classic_fwd,
            'mem_reduc_compile_fwd': mem_reduc_compile_fwd,
            'mem_reduc_classic_bwd': mem_reduc_classic_bwd,
            'mem_reduc_compile_bwd': mem_reduc_compile_bwd,
        }
    return results

def plot_memory_results(results, filename, mode='fwd'):
    """Plot memory comparison results."""
    assert mode in ['fwd', 'bwd']
    if not results:
        print("No results to plot.")
        return
        
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    seq_lens = sorted(list(set(key for key in results.keys())))
    
    # Aggregate results by sequence length
    mem_reduc_classic, mem_reduc_compile = [], []
    classic_mem, compile_mem, flash_mem = [], [], []

    for seq_len in seq_lens:
        mem_reduc_classic.append(np.mean([v[f'mem_reduc_classic_{mode}'] for k, v in results.items() if k == seq_len]))
        mem_reduc_compile.append(np.mean([v[f'mem_reduc_compile_{mode}'] for k, v in results.items() if k == seq_len]))
        classic_mem.append(np.mean([v[f'classic_{mode}_mem'][0] for k, v in results.items() if k == seq_len]))
        compile_mem.append(np.mean([v[f'compile_{mode}_mem'][0] for k, v in results.items() if k == seq_len]))
        flash_mem.append(np.mean([v[f'flash_{mode}_mem'][0] for k, v in results.items() if k == seq_len]))

    # Plot memory reductions
    axs[0].plot(seq_lens, mem_reduc_classic, 'o-', label=f'vs Classic {mode}')
    axs[0].plot(seq_lens, mem_reduc_compile, 's-', label=f'vs Compile {mode}')
    axs[0].set_xlabel('Sequence Length')
    axs[0].set_ylabel('Memory Reduction (x)')
    axs[0].set_title('Flash Attention Memory Efficiency vs. Classic & Compile')
    axs[0].legend()
    axs[0].grid(True)

    # Plot absolute memory usage
    axs[1].plot(seq_lens, classic_mem, 'o-', color='blue', label=f'Classic {mode}')
    axs[1].plot(seq_lens, compile_mem, 's-', color='red', label=f'Compile {mode}')
    axs[1].plot(seq_lens, flash_mem, '^-', color='green', label=f'Flash {mode}')

    axs[1].set_xlabel('Sequence Length')
    axs[1].set_ylabel('Peak Memory Usage (MB)')
    axs[1].set_title('Memory Usage Comparison')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(filename + '.png')
    plt.show()

def save_results_to_csv(results, filename, mode='fwd'):
    """Saves the benchmark results to a CSV file."""
    if not results:
        print("No results to save.")
        return
    
    header = [
        'seq_len',
        f'classic {mode} memory (MB)',
        f'compile {mode} memory (MB)',
        f'flash {mode} memory (MB)',
        f'{mode} memory reduction vs classic',
        f'{mode} memory reduction vs compile',
    ]

    try:
        with open(filename + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header) # Write the header row
            # Write data rows, sorted by sequence length
            for seq_len in sorted(results.keys()):
                data = results[seq_len]
                row = [
                    seq_len,
                    f"{data[f'classic_{mode}_mem'][0]:.2f}",
                    f"{data[f'compile_{mode}_mem'][0]:.2f}",
                    f"{data[f'flash_{mode}_mem'][0]:.2f}",
                    f"{data[f'mem_reduc_classic_{mode}']:.2f}",
                    f"{data[f'mem_reduc_compile_{mode}']:.2f}"
                ]
                writer.writerow(row)
    except IOError as e:
        print(f"Error saving file: {e}")

def run_memory_benchmarks():
    """Run all benchmarks with a focus on memory allocation."""
    # Parameters for tests
    batch_size = 16
    num_windows = 4
    num_heads = 6
    head_dim = 32
    
    seq_lens = [8**2, 12**2, 16**2, 24**2, 32**2, 40**2]
    

    print("\n=== Running Memory Allocation Benchmark ===")
    results = benchmark_memory_allocation(
        seq_lens,
        batch_size,
        num_windows,
        num_heads,
        head_dim,
    )

    if results:
        out_dir = "benchmark/results"
        os.makedirs(out_dir, exist_ok=True)
        
        # Save results to CSV and plot
        for mode in ["fwd", "bwd"]:
            filename = os.path.join(out_dir, f"win-attention-memory-comparaison-batch{batch_size}-window{num_windows}-head{num_heads}-d{head_dim}-{mode}")
            save_results_to_csv(results, filename, mode)
            plot_memory_results(results, filename, mode)


if __name__ == "__main__":
    run_memory_benchmarks()