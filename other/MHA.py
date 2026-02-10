import torch
import torch.nn as nn
import math
from torch import Tensor

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, head_nums, drop_rate=0.03):
        super(MultiHeadAttention).__init__()
        self.head_nums = head_nums

        self.dim = emb_dim // head_nums  # 必须整数

        self.proj = nn.Linear(emb_dim, 3*emb_dim)
        
        self.o_proj = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(drop_rate)

    def gen_mask(self, seq_):
        # seq_ [B, seq]
        # return full-attn padding mask
        return None

    
    def forward(self, X:Tensor, mask):
        # X [batchsize, seq_len, emb_dim]
        batchsize, seq_len, dim = X.shape
        qkv:Tensor = self.proj(X)
        # [3, B, headnum, seq, dim]
        qkv = qkv.view(batchsize, seq_len,3, self.head_nums, -1).permute(2, 0, 3, 1, 4) # [batchsize, seq_len, emb_dim,3]
        Q, K, V = qkv.unbind(0)

        Q:Tensor
        K:Tensor
        V:Tensor

        
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dim)

        if mask is not None: # [batchsize, 1, seq_len, seq_len]   # 如果是因果注意力机制，应该是个下三角矩阵
            # 加下掩码 形状一样，直接广播乘法
            attention_scores = attention_scores.masked_fill(mask==0.0, -math.inf)


        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        
        # [batchsize, head_nums, seq, dim]
        vectors = torch.matmul(attention_scores, V)
        # vectors = vectors.transpose(1, 2).contiguous().view(batchsize, self.seq_len, -1)
        vectors = vectors.transpose(1, 2).reshape(batchsize, seq_len, -1)

        return self.o_proj(vectors)


