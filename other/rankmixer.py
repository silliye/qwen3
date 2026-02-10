
import torch
from torch import Tensor

import torch.nn as nn

class RankMixer(nn.Module):
    def __init__(self, token_nums, token_dim, head_nums, intervel_dim, layers_num):
        super().__init__()
        self.token_nums = token_nums
        self.token_dim = token_dim
        self.head_nums = head_nums

        self.intervel_dim = intervel_dim
        self.layers_nums = layers_num

        self.rankmixers = nn.ModuleList(RankMixerLayer(token_nums, token_dim, head_nums, intervel_dim) for _ in range(layers_num))
        
    def forward(self, x):
        out = x
        for i in range(self.layers_nums):
            out = self.rankmixers[i](out)
        return out
    
class FFN(nn.Module):
    def __init__(self, input_dim, intervel_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, intervel_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(intervel_dim, input_dim)
    def forward(self, x):
        return self.layer2(self.gelu(self.layer1(x)))

class RankMixerLayer(nn.Module):
    def __init__(self, per_token_expert_nums, token_nums, token_dim, head_nums, intervel_dim):
        super(RankMixerLayer).__init__()
        self.token_nums = token_nums
        self.token_dim = token_dim
        self.head_nums = head_nums
        self.dim = token_dim // head_nums

        self.intervel_dim = intervel_dim

        self.per_token_expert_nums = per_token_expert_nums
    
        # FFN
        self.gelu = nn.GELU()

        self.perFFN_layer1_w = nn.Parameter(torch.Tensor(token_nums, token_dim, intervel_dim))
        self.perFFN_layer1_b = nn.Parameter(torch.Tensor(token_nums, intervel_dim))

        
        self.perFFN_layer2_w = nn.Parameter(torch.Tensor(token_nums, intervel_dim, token_dim))
        self.perFFN_layer2_b = nn.Parameter(torch.Tensor(token_nums, token_dim))

        # norm
        self.layerNorm_attention = nn.LayerNorm(token_dim)

        self.layerNorm_ffn = nn.LayerNorm(token_dim)


        

        self.para_init()

    def para_init(self):
        nn.init.kaiming_uniform_(self.perFFN_layer1_w)
        nn.init.kaiming_uniform_(self.perFFN_layer2_w)

        nn.init.zeros_(self.perFFN_layer1_b)
        nn.init.zeros_(self.perFFN_layer2_b)



    def mixer(self, input:Tensor):
        # input [B, T, D]
        B, T, D = input.shape
        # [B, T, heads, D//heads] -> [B, heads, T, D//heads]
        return input.view(B, T, self.head_nums, self.dim).transpose(1, 2).contiguous().view(B, self.head_nums, -1)
        

    def attn_norm(self, input):
        # axis = -1
        return self.layerNorm_attention(input)

    def ffn_norm(self, input):
        # axis = -1
        return self.layerNorm_ffn(input)
    
    def pffn(self, input:Tensor):
        # [B, T, D]

        # [B, T, 1, D] * [1, T, D, MD] + [T, MD] = [B, T, 1, MD] + [T, MD]
        hidden_dim = input.unsqueeze(-2) @ self.perFFN_layer1_w.unsqueeze(0) + self.perFFN_layer1_b.unsqueeze(0).unsqueeze(-2)
        hidden_dim = self.gelu(hidden_dim)
        # [B, T, 1, MD] * [1, T, MD, D] + [T, D] = [B, T, 1, D]
        output_dim = hidden_dim @ self.perFFN_layer2_w.unsqueeze(0) + self.perFFN_layer2_b.unsqueeze(0).unsqueeze(-2)
        return output_dim.squeeze(-2)

    def forward(self, input):
        # input [B, T, Dim]
        
        # postNorm

        mixer_output = self.mixer(input)
        ffn_input = self.attn_norm(mixer_output + input)

        return self.ffn_norm(self.pffn(ffn_input) + ffn_input)