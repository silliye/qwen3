
import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F

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

        # norm
        self.layerNorm_attention = nn.LayerNorm(token_dim)

        self.layerNorm_ffn = nn.LayerNorm(token_dim)

         # FFN
        self.gelu = nn.GELU()

        # gated [Token_num, token_dim, experts_num]
        self.train_router = nn.Parameter(Tensor(token_nums, token_dim, per_token_expert_nums))
        self.infer_router = nn.Parameter(Tensor(token_nums, token_dim, per_token_expert_nums))


        self.expertsFFN_layer1_w = nn.Parameter(torch.Tensor(token_nums, per_token_expert_nums, token_dim, intervel_dim))
        self.expertsFFN_layer1_b = nn.Parameter(torch.Tensor(token_nums, per_token_expert_nums, intervel_dim))

        
        self.expertsFFN_layer2_w = nn.Parameter(torch.Tensor(token_nums, per_token_expert_nums, intervel_dim, token_dim))
        self.expertsFFN_layer2_b = nn.Parameter(torch.Tensor(token_nums, per_token_expert_nums, token_dim))


        self.para_init()

    def para_init(self):
        nn.init.kaiming_uniform_(self.expertsFFN_layer1_w)
        nn.init.kaiming_uniform_(self.expertsFFN_layer2_w)

        nn.init.zeros_(self.expertsFFN_layer1_b)
        nn.init.zeros_(self.expertsFFN_layer2_b)



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

    def sparse_moe(self, input:Tensor, therehold=0):
        # input [B, T, D]
        B, T, D = input.shape

        # [B, T, 1, per_token_expert_nums]
        train_router = torch.relu(input.unsqueeze(-2) @ self.train_router.unsqueeze(0)).squeeze(-2)
        infer_router = torch.relu(input.unsqueeze(-2) @ self.train_router.unsqueeze(0)).squeeze(-2)

        # bool Tensor [B, T, per_token_expert_nums]
        train_router_bigger_therehold = train_router > therehold 

        # indices Tensor [N, 3]  这个N是非
        train_router_bigger_therehold_indices = torch.nonzero(train_router_bigger_therehold)

        '''
                tensor([
        [ # Batch 0
            [False, True,  False],  # Token 0: 激活了 Expert 1
            [True,  False, True ]   # Token 1: 激活了 Expert 0 和 2
        ],
        [ # Batch 1
            [False, False, False],  # Token 0: 无激活
            [False, True,  True]    # Token 1: 激活了 Expert 1,2
        ]
        ])

        tensor([
            [0, 0, 1],  # Batch 0, Token 0, Expert 1
            [0, 1, 0],  # Batch 0, Token 1, Expert 0
            [0, 1, 2],  # Batch 0, Token 1, Expert 2
            [1, 1, 1]   # Batch 1, Token 1, Expert 1
            [1, 1, 2]
            ])
        
        '''

        # [0, 0, 0, 1, 1]
        batch_idx = train_router_bigger_therehold_indices[:, 0]
        # [0, 1, 1, 1. 1]
        token_idx = train_router_bigger_therehold_indices[:, 1]
        expert_idx = train_router_bigger_therehold_indices[:, 2]

        # 高级索引 [0, 0] [0, 1] [0, 1] [1, 1] [1, 1]

        # [5, D] 取要计算的输入 (这里相当于 把要计算的 输入都拿过来，5就是有被激活的专家总数，这里可能会有重复，因为这是batch-token维度，还有多个expert要被激活)
        input_active = input[batch_idx, token_idx]
        


        




    def forward(self, input):
        # input [B, T, Dim]
        
        # postNorm

        mixer_output = self.mixer(input)
        ffn_input = self.attn_norm(mixer_output + input)

        return self.ffn_norm(self.pffn(ffn_input) + ffn_input)