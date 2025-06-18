import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import measure_speed_memory, save_results_to_csv
from config import swin_transformer_config
from flash_win_attn_v2 import SwinTransformerV2


@torch.inference_mode
def forward(batch_size: int, config: dict, dtype=torch.float16):
    x = torch.randn(batch_size, config["in_chans"], config["img_size"], config["img_size"], dtype=dtype).cuda()
    m = SwinTransformerV2(**config).to(dtype=dtype, device='cuda')
    t, memory = measure_speed_memory(m.forward, x)

    return t, memory


def backward(batch_size: int, config: dict, dtype=torch.float16):
    x = torch.randn(batch_size, config["in_chans"], config["img_size"], config["img_size"], dtype=dtype).cuda()
    y = torch.randint(0, 1000, size=(batch_size,)).cuda()
    m = SwinTransformerV2(**config).to(dtype=dtype, device='cuda')
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        
    def _f(m, opt, x, y):
        opt.zero_grad()
        o = m.forward(x)
        loss = F.cross_entropy(o, y)
        loss.backward()
        opt.step()

    t, memory = measure_speed_memory(_f, m, opt, x, y)

    return t, memory


def run_model_benchmarks():
    batch_sizes = [8, 16, 32, 64]
    dtype = torch.float16
    
    flash_config = swin_transformer_config.copy()
    flash_config.update({"is_flash": True})
    classic_config = swin_transformer_config.copy()
    classic_config.update({"is_flash": False})
    
    # Storage for results
    results = {
        'batch_sizes': batch_sizes,
        'forward': {
            'classic_time': [], 'classic_memory': [],
            'flash_time': [], 'flash_memory': []
        },
        'backward': {
            'classic_time': [], 'classic_memory': [],
            'flash_time': [], 'flash_memory': []
        }
    }
    
    print("Running benchmarks...")
    for batch in batch_sizes:
        print(f"Testing batch size: {batch}")
        
        # Forward pass benchmarks
        t, memory = forward(batch, classic_config, dtype=dtype)
        results['forward']['classic_time'].append(t)
        results['forward']['classic_memory'].append(memory)
        print(f"  Forward classic: {t:.3f}ms, {memory:.3f}GB")
        
        t, memory = forward(batch, flash_config, dtype=dtype)
        results['forward']['flash_time'].append(t)
        results['forward']['flash_memory'].append(memory)
        print(f"  Forward flash: {t:.3f}ms, {memory:.3f}GB")
        
        # Backward pass benchmarks
        t, memory = backward(batch, classic_config, dtype=dtype)
        results['backward']['classic_time'].append(t)
        results['backward']['classic_memory'].append(memory)
        print(f"  Backward classic: {t:.3f}ms, {memory:.3f}GB")
        
        t, memory = backward(batch, flash_config, dtype=dtype)
        results['backward']['flash_time'].append(t)
        results['backward']['flash_memory'].append(memory)
        print(f"  Backward flash: {t:.3f}ms, {memory:.3f}GB")
        
        # Clear GPU cache between batches
        torch.cuda.empty_cache()
    
    save_results_to_csv(results, swin_transformer_config)

    return results

if __name__ == '__main__':
    print("\nRunning Swin Transformer V2 benchmarks...")
    print("-" * 50)
    run_model_benchmarks()
    