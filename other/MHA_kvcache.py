import torch
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, head_nums, drop_rate=0.03):
        super().__init__()
        self.head_nums = head_nums

        self.dim = emb_dim // head_nums  # 必须整数

        self.qkvproj = nn.Linear(emb_dim, 3*emb_dim)
        
        self.o_proj = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(drop_rate)


    def gen_padding_mask(self, mask):
        # [B, seq]
        B, seq = mask.shape
        ones = torch.ones([B, 1], device=mask.device)

        # [B, seq+1]
        mask = torch.concat([mask, ones], dim=-1)

        # [B, 1, 1, seq+1]
        return mask, mask.unsqueeze(1).unsqueeze(1)
    

    def forward(self, X:Tensor, kcache:Tensor, vcache:Tensor, mask):
        # left padding : kcache; vcache [B, seq, dim]
        # mask [B, seq]
        # mask = [[0, 0, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1]]

        # X [batchsize, 1, emb_dim]
        batchsize, seq_len, dim = X.shape

        qkv:Tensor = self.qkvproj(X)

        # [3, B, headnum, 1, dim]
        qkv = qkv.view(batchsize, seq_len,3, self.head_nums, -1).permute(2, 0, 3, 1, 4) # [batchsize, seq_len, emb_dim,3]
        Q, K, V = qkv.unbind(0)

        if not kcache:
            kcache = K
        else:
            # [B, headnum, seq+1, dim]
            kcache = torch.cat([kcache, K], dim=-2)
        if not vcache:
            vcache = V
        else:
            vcache = torch.cat([vcache, V], dim=-2)

        Q:Tensor = Q
        K:Tensor = kcache
        V:Tensor = vcache

        # Q [B, head, 1, dim]
        # KV [B, head, seq+1, dim]

        mask, padding_mask = self.gen_padding_mask(mask)

        # [B, head, 1, seq+1]
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.dim)
        
        if padding_mask is not None:
            attention_scores.masked_fill_(padding_mask == 0, float('-inf'))


        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        
        # [batchsize, head_nums, seq, dim]
        vectors = torch.matmul(attention_scores, V)
        # vectors = vectors.transpose(1, 2).contiguous().view(batchsize, self.seq_len, -1)
        vectors = vectors.transpose(1, 2).contiguous().view(batchsize, seq_len, -1)

        return self.o_proj(vectors), kcache, vcache, mask


class FFN(nn.model):
    def __init__(self, input_dim, intermediate_dim):
        super().__init__()

        self.gate = nn.Linear(input_dim, intermediate_dim)
        self.up = nn.Linear(input_dim, intermediate_dim)
        self.down = nn.Linear(intermediate_dim, input_dim)
        self.silu = nn.SiLU()
    
    def forward(self, X):
        return self.down(self.silu(self.gate(X)) * self.up(X))

class RMSNorm(nn.Module):

    def __init__(self, input_dim):
        super.__init__()

        self.weight = nn.Parameter(torch.ones(input_dim))
    
    def forward(self, X):
        # X [B, seq, dim]
        eposilon = 1e-5
        X = X / torch.sqrt(torch.mean(torch.pow(X, 2), dim=-1, keepdim=True) + eposilon)

        return X * self.weight


class DecoderLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, head_nums):
        super().__init__()

        self.attention_layer = MultiHeadAttention(input_dim, head_nums)
        self.ffn_layer = FFN(input_dim, intermediate_dim)

        self.attn_rmsnorm = RMSNorm(input_dim)
        self.ffn_rmsnorm = RMSNorm(input_dim)

    
    def forward(self, X, kcache, vcache, mask):
        attention_input = self.attn_rmsnorm(X)
        attention_output, kcache, vcache, mask = self.attention_layer(attention_input, kcache, vcache, mask) + X

        ffn_input = self.ffn_rmsnorm(attention_output)

        ffn_output = self.ffn_layer(ffn_input) + attention_output

        return ffn_output, kcache, vcache, mask 


class LLM(nn.Module):
    def __init__(self, input_dim, intermediate_dim, head_nums, vocab_size, decoder_layers):
        super().__init__()

        self.decoders = nn.ModuleList([DecoderLayer] * (decoder_layers))

        self.vocab_emb = nn.Embedding(vocab_size, input_dim)

        self.decoder_layers = decoder_layers
        self.bos_token = 10043
        self.eos_token = 10044  

        self.input_norm = RMSNorm(input_dim)


        
    # eval:
    def forward(self):
        batchsize = 1000
        # [B, 1] bos -> 10037
        bos = torch.ones([batchsize, 1]) * self.bos_token

        # [B, 1]
        tokens = bos
        # 这里emb维度的细节需要注意一下
        kcache = [None] * self.decoder_layers
        vcache = [None] * self.decoder_layers
        mask = None
        while True:
            # [B, 1, emb]
            X = self.vocab_emb(tokens[:, -1])

            X = self.input_norm(X)

            for _ in range(self.decoder_layers):
                
                X, kcache[_], vcache[_], mask  = self.decoders[_](X, kcache[_], vcache[_], mask )
            
            # [B, 1, vocab]
            vocab_ = X[:, -1, :] @ self.vocab_emb.weight.T

            # [B, 1]
            vocab_token_new = torch.argmax(vocab_, dim=-1)

            # [B, 2]
            tokens = torch.concat([tokens, vocab_token_new])

        





