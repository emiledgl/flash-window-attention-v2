import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


from flash_win_attn_v2 import FlashWindowAttentionV2

flash_win_attn_v2 = FlashWindowAttentionV2.apply

def win_attention_torch(
    q: torch.Tensor, # (batch_size * windows, h, seq, head_dim)
    k: torch.Tensor, # (batch_size * windows, h, seq, head_dim)
    v: torch.Tensor, # (batch_size * windows, h, seq, head_dim)
    logit_scale: torch.Tensor = None,
    bias: torch.Tensor = None, # (h, seq, seq)
    mask: torch.Tensor = None, # (windows, seq, seq)
    ) -> torch.Tensor:

    bw_size, n_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape
    q_norm = F.normalize(q, p=2, dim=-1) # (batch_size * windows, h, seq, head_dim)
    k_norm = F.normalize(k, p=2, dim=-1) # (batch_size * windows, h, seq, head_dim)
    qk = torch.matmul(q_norm, k_norm.transpose(-2, -1)) # (batch_size * windows, h, seq, seq)
    if logit_scale is not None:
        logit_scale = logit_scale.clamp(max=torch.log(torch.tensor(100.0, dtype=logit_scale.dtype, device=logit_scale.device))).exp()
    else:
        logit_scale = 1.0 / torch.sqrt(torch.tensor(head_dim, device=q.device))
    attn = qk * logit_scale # (batch_size * windows, h, seq, seq)
    if bias is not None:
        attn += bias
    if mask is not None:
        windows = mask.shape[0]
        attn = attn.view(bw_size // windows, windows, n_heads, seq_len_q, seq_len_k) # (batch, windows, h, seq, seq)
        attn += mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, n_heads, seq_len_q, seq_len_k)
    attn = torch.softmax(attn, dim=-1)
    output = torch.matmul(attn, v) # (batch_size * windows, h, seq, head_dim)
    return output


@torch.compile
def win_attention_torch_compile(
    q: torch.Tensor, # (batch_size * windows, h, seq, head_dim)
    k: torch.Tensor, # (batch_size * windows, h, seq, head_dim)
    v: torch.Tensor, # (batch_size * windows, h, seq, head_dim)
    logit_scale: torch.Tensor = None,
    bias: torch.Tensor = None, # (h, seq, seq)
    mask: torch.Tensor = None, # (windows, seq, seq)
    ) -> torch.Tensor:

    bw_size, n_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape
    q_norm = F.normalize(q, p=2, dim=-1) # (batch_size * windows, h, seq, head_dim)
    k_norm = F.normalize(k, p=2, dim=-1) # (batch_size * windows, h, seq, head_dim)
    qk = torch.matmul(q_norm, k_norm.transpose(-2, -1)) # (batch_size * windows, h, seq, seq)
    if logit_scale is not None:
        logit_scale = logit_scale.clamp(max=torch.log(torch.tensor(100.0, dtype=logit_scale.dtype, device=logit_scale.device))).exp()
    else:
        logit_scale = 1.0 / torch.sqrt(torch.tensor(head_dim, device=q.device))
    attn = qk * logit_scale # (batch_size * windows, h, seq, seq)
    if bias is not None:
        attn += bias
    if mask is not None:
        windows = mask.shape[0]
        attn = attn.view(bw_size // windows, windows, n_heads, seq_len_q, seq_len_k) # (batch, windows, h, seq, seq)
        attn += mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, n_heads, seq_len_q, seq_len_k)
    attn = torch.softmax(attn, dim=-1)
    output = torch.matmul(attn, v) # (batch_size * windows, h, seq, head_dim)
    return output