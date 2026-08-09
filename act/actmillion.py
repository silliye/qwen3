import torch

import torch.nn as nn

from torch import Tensor

def register():
    # 注册FA 模拟
    return 

class SwishGLUFFN(nn.Module):
    def __init__(self, tokens_dim, intermiddle_token_dim):
        super().__init__()

        self.tokens_dim = tokens_dim
        self.intermiddle_token_dim = intermiddle_token_dim

        self.up_proj = nn.Linear(tokens_dim, intermiddle_token_dim)
        self.gate_proj = nn.Linear(tokens_dim, intermiddle_token_dim)
        self.down_proj = nn.Linear(intermiddle_token_dim, tokens_dim)
        self.swish = torch.nn.SiLU()
    def forward(self, X):
        up = self.up_proj(X)
        gate = self.gate_proj(X)
    
        return self.down_proj(up * self.swish(gate))



class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()

        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, X):
        # X [H, dim]
        rms = torch.sqrt(torch.mean(torch.pow(X, 2), dim=-1,keepdim=True) + self.eps) 
        X = X / rms * self.weight

        return X
        

class ACTMillionBlock(nn.Module):
    def __init__(self, query_tokens_dim, kv_input_dim, head_dim, head_num, intermiddle_token_dim):
        super().__init__()

        self.query_tokens_dim = query_tokens_dim
        self.kv_input_dim = kv_input_dim
        # FFN
        self.intermiddle_token_dim = intermiddle_token_dim

        # attn:
        self.head_num = head_num
        self.head_dim = head_dim
        self.attn_rmsnorm = RMSNorm(query_tokens_dim)

        self.ffn_rmsnorm = RMSNorm(query_tokens_dim)

        self.ffn = SwishGLUFFN(query_tokens_dim, intermiddle_token_dim)
        self.q_proj = nn.Linear(query_tokens_dim, head_dim*head_num)
        self.kv_proj = nn.Linear(kv_input_dim, head_dim*head_num*2)

        self.varlenFlashAttn = register()

    def forward(self, query_input, kv_input, q_seq_info, kv_seq_info):
        query_input = self.attn_rmsnorm(query_input)

        usernumsquery_token_num, query_dim =  query_input.shape
        kv_seq, kv_dim = kv_input.shape
        # [usernum, 500, 256] -> [4, usernum * 500, 64]
        
        querys = self.q_proj(query_input).view(usernumsquery_token_num, self.head_num, self.head_dim).transpose(0, 1)
        # [H, 256*2]
        kv = self.kv_proj(kv_input).view(kv_seq, 2, self.head_num, self.head_dim).permute(1, 2, 0, 3)
        # [4, kv_seq, 64]
        keys, values = kv.unbind(0)

        # 自定义顺序, head放第0维度
        attn_output = self.varlenFlashAttn(querys, keys, values, 
                                      q_seq_info = q_seq_info,
                                      kv_seq_info = kv_seq_info,
                                      bf16 = True,
                                      mask = 'BlockDiagalMask')
        attn_output = attn_output + query_input
        ffn_input = self.ffn_rmsnorm(attn_output)

        ffn_output = self.ffn(ffn_input) + attn_output

        return ffn_output
        

        

class ACTMillion(nn.Module):
    def __init__(self, query_tokens_num, query_tokens_dim, input_dim, kv_input_dim, nums_layer, intermiddle_token_dim, head_dim, head_num):
        super().__init__()

        self.query_tokens_num = query_tokens_num
        self.query_tokens_dim = query_tokens_dim
        self.input_dim = input_dim
        self.kv_input_dim = kv_input_dim
        self.nums_layer = nums_layer
        # FFN
        self.intermiddle_token_dim = intermiddle_token_dim

        # attn:
        self.head_num = head_num
        self.head_dim = head_dim
        # [500, 256]
        self.anchor_querys = nn.Parameter(torch.rand(query_tokens_num, query_tokens_dim))
        # [216, 256]
        self.feature_linear = nn.Linear(input_dim, kv_input_dim)

        self.ACTmillionBlocks = nn.ModuleList([ACTMillionBlock(query_tokens_dim, kv_input_dim,head_dim, head_num, intermiddle_token_dim) for _ in range(nums_layer)])
        

    def forward(self, X_input, colossus_mask):
        # X_input : [usernum, 10w, dim216]
        # colossus_mask : [usernum, 10w]
        usernum, max_seq_len, dim = X_input.shape


        colossus_mask = colossus_mask > 0
        single_user_seq_sum = torch.sum(colossus_mask, dim=-1)

        # [usernum, 10w, dim] [usernum, 10w bool] -> [token, dim]
        # [effect_tokens, dim]
        X_input = X_input[colossus_mask]
        kv_input = self.feature_linear(X_input)

        
        interest_token = self.anchor_querys.unsqueeze(0).expand(usernum, -1, -1).view(-1, self.query_tokens_dim)
        q_seq_info = [self.query_tokens_num] * usernum
        kv_seq_info = single_user_seq_sum # [usernum, seq]
        for i in range(self.nums_layer):
            
            interest_token = self.ACTmillionBlocks[i](interest_token, kv_input, q_seq_info, kv_seq_info)

        return interest_token
        
        

