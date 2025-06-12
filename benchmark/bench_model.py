import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import measure_speed_memory
from flash_win_attn_v2 import SwinTransformerV2


@torch.inference_mode
def forward(batch, is_flash, dtype=torch.float16):
    x = torch.randn(batch, 3, 256, 256, dtype=dtype).cuda()
    m = SwinTransformerV2(is_flash=is_flash).to(dtype=dtype, device='cuda')
    t, memory = measure_speed_memory(m.forward, x)

    return t, memory


def backward(batch, is_flash, dtype=torch.float16):
    x = torch.randn(batch, 3, 256, 256, dtype=dtype).cuda()
    y = torch.randint(0, 1000, size=(batch,)).cuda()
    m = SwinTransformerV2(is_flash=is_flash).to(dtype=dtype, device='cuda')
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        
    def _f(m, opt, x, y):
        opt.zero_grad()
        o = m.forward(x)
        loss = F.cross_entropy(o, y)
        loss.backward()
        opt.step()

    t, memory = measure_speed_memory(_f, m, opt, x, y)

    return t, memory


if __name__ == '__main__':
    
    batch = [4, 8, 16, 32, 64]
    dtype=torch.float16

    for _batch in batch:
        print(f"batch={_batch}")
        t, memory = forward(_batch, is_flash=False, dtype=dtype)
        print(f"forward; flash=OFF:  {t:.3f}ms {memory:.3f}GB")
        t, memory = forward(_batch, is_flash=True, dtype=dtype)
        print(f"forward; flash=ON:   {t:.3f}ms {memory:.3f}GB")
        t, memory = backward(_batch, is_flash=False, dtype=dtype)
        print(f"backward; flash=OFF: {t:.3f}ms {memory:.3f}GB")
        t, memory = backward(_batch, is_flash=True, dtype=dtype)
        print(f"backward; flash=ON:  {t:.3f}ms {memory:.3f}GB")