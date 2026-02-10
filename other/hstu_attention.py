import torch
import torch.nn as nn
from torch import Tensor
import math

class HSTUAttention(nn.Module):
    def __init__(self, dim, head_nums):
        super().__init__()

        self.head_nums = head_nums
        self.dim = dim

        self.Linear1 = nn.Linear(dim, 4*dim)
        self.silu = nn.SiLU()

        self.layernorm = nn.LayerNorm(dim)

        self.Linear2 = nn.Linear(dim, dim)

    

    def forward(self, x:Tensor):
        # x: [B, seq, dim]
        B, seq, dim = x.shape

        # [B, seq, 4, head, dim // head] - > [4, B, head, seq, dim // head]
        gate, q, k, v = self.silu(self.Linear1(x)).view(B, seq, 4, self.head_nums, dim // self.head_nums).permute(2, 0, 3, 1, 4).contiguous().unbind(0)
        

        # 就不考虑padding了:  [B, seq] padding mask
        # 

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        casual_mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()

        # [B, head, seq, seq]
        attention_scores:Tensor = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim // self.head_nums)

        attention_scores.masked_fill_(casual_mask == 1.0, -torch.inf)

        # [B, head, seq, dim//head] -> [B, seqm dim]
        normed = self.layernorm(torch.matmul(self.silu(attention_scores), v).transpose(1, 2).contiguous().view(B, seq, -1))

        # gate [B, seq, head, d // h]
        gate = gate.transpose(1, 2).contiguous().view(B, seq, -1)

        return self.Linear2(normed * gate)



