import torch
import torch.nn as nn
import math
from torch import Tensor

class Decoder(nn.Module):
    def __init__(self, emb_dim, head_nums, max_seq_length=4096):
        super().__init__()
        # [seq, dim]
        self.head_nums = head_nums
        self.register_buffer('cos_emb', tensor=torch.empty(1, emb_dim))
        self.register_buffer('sin_emb', tensor=torch.empty(1, emb_dim))
        self.max_seq_length = 0
        self.attention_layers = MultiHeadAttention(emb_dim, head_nums)

    def get_rope_emb(self, X:Tensor):
        # X [B, seq, dim]
        _, seq_length, dim = X.shape
        if seq_length < self.max_seq_length:
            return self.cos_emb[:seq_length], self.sin_emb[:seq_length]
        else:
            self.max_seq_length = seq_length
            # [dim // 2]
            angle = 1 / torch.pow(10000, torch.arange(0, dim // self.head_nums, 2) / dim)
            # [seq]
            seq = torch.arange(0, self.max_seq_length)
            # [seq, dim//2]
            rope_emb = seq.unsqueeze(1) @ angle.unsqueeze(0)
            # [seq, dim]
            rope_emb = torch.repeat_interleave(rope_emb, repeats=2, dim=-1)
            # [seq, dim]
            self.cos_emb = torch.cos(rope_emb)
            self.sin_emb = torch.sin(rope_emb)

            return self.cos_emb, self.sin_emb

        
    def forward(self, X):
        
        cos_emb, sin_emb = self.get_rope_emb(X)

        self.attention_layers(X, mask=None, cos_emb=cos_emb, sin_emb=sin_emb)

    

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, head_nums, drop_rate=0.03):
        super().__init__()
        self.head_nums = head_nums

        self.dim = emb_dim // head_nums  # 必须整数

        self.proj = nn.Linear(emb_dim, 3*emb_dim)
        
        self.o_proj = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(drop_rate)
    
   

    def gen_casual_mask(self, mask:Tensor):
        # mask [B, seq] [1, 1, 1, 0]
        batch, seq = mask.shape

        # [B, 1, 1, seq]
        pad_mask = mask.unsqueeze(1).unsqueeze(1)

        # [1, 1, seq, seq]
        casual_mask = torch.tril(torch.ones(seq, seq, device=mask.device)).unsqueeze(0).unsqueeze(0)

        
        return (pad_mask * casual_mask)

    
    
    def gen_full_mask(self, mask:Tensor):
        # mask [B, seq] [1, 1, 1, 0]

        # [B, 1, 1, seq]
        pad_mask = mask.unsqueeze(1).unsqueeze(1)

        # [B, 1, 1, seq]
        return pad_mask

    def apply_rope(self, query, key, cos_emb, sin_emb):
        # query [B, head, seq, dim]
        # key [B, head, seq, dim]
        # cos [seq, dim]
        # sin [seq, dim]

        def trans(X):
            # X [B, head, seq, dim]
            B, head, seq, dim = X.shape
            X_trans = X.view(B, head, seq, dim // 2, 2)
            X1, X2 = X_trans[:,:,:,:,0], X_trans[:,:,:,:,1]
            return torch.concat([-X2, X1], dim=-1).reshape([B, head, seq, dim])

        return cos_emb.unsqueeze(0).unsqueeze(1) * query + sin_emb.unsqueeze(0).unsqueeze(1) * trans(query), cos_emb.unsqueeze(0).unsqueeze(1) * key + sin_emb.unsqueeze(0).unsqueeze(1) * trans(key)


    def forward(self, X:Tensor, mask, cos_emb, sin_emb):
        # X [batchsize, seq_len, emb_dim]
        batchsize, seq_len, dim = X.shape
        qkv:Tensor = self.proj(X)
        # [3, B, headnum, seq, dim]
        qkv = qkv.view(batchsize, seq_len, 3, self.head_nums, -1).permute(2, 0, 3, 1, 4) # [3, batchsize, headnums, seq_len, dim]
        Q, K, V = qkv.unbind(0)

        # [batchsize, head_nums, seq_len, dim]
        Q:Tensor
        K:Tensor
        V:Tensor
        Q, K = self.apply_rope(Q, K, cos_emb, sin_emb)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dim)

        
        # casual_pad_mask = self.gen_casual_mask(mask)
        padding_mask = self.gen_full_mask(mask)
        

        if padding_mask is not None: # [batchsize, 1, seq_len, seq_len]   # 如果是因果注意力机制，应该是个下三角矩阵
            # 加下掩码 形状一样，直接广播乘法
            attention_scores = attention_scores.masked_fill_(padding_mask==0.0, -1e9)


        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        
        # [batchsize, head_nums, seq, dim]
        vectors = torch.matmul(attention_scores, V)
        # vectors = vectors.transpose(1, 2).contiguous().view(batchsize, self.seq_len, -1)
        vectors = vectors.transpose(1, 2).reshape(batchsize, seq_len, -1)

        return self.o_proj(vectors)


