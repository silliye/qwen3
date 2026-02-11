import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
import math
import numpy as np

# ROPE

class BaseNetwork(nn.Module):
    def __init__(self, name=""):
        super(BaseNetwork, self).__init__()
        self.name = name
    
    def extra_repr(self):
        return f'name={self.name}'

class MLP(BaseNetwork):
    def __init__(self, input_dim, hidden_dims, output_dim=None, activation="RELU", if_act_last=False, name="mlp"):
        super().__init__(name)
        layers = []
        
        if activation == "RELU":
            act_layer = nn.ReLU
        elif activation == "GELU":
            act_layer = nn.GELU
        elif activation == "Sigmoid":
            act_layer = nn.Sigmoid
        else:
            act_layer = nn.ReLU

        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_layer())
            prev_dim = h_dim
        
        if output_dim:
            layers.append(nn.Linear(prev_dim, output_dim))
        
        if if_act_last:
            layers.append(act_layer())
        
        self.net = nn.Sequential(*layers)



    def forward(self, x):
        return self.net(x)

class RMSNorm(BaseNetwork):
    def __init__(self, dim, eps, name="RMSNorm"):
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        super().__init__(name)
    
    def _norm(self, x:Tensor):
        rms = torch.sqrt(torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps)
        return x / rms
    
    def forward(self, x:Tensor):
        return self._norm(x) * self.weight

class Qwen2EncoderLayer(BaseNetwork):
    def __init__(self, hidden_dim, num_attention_heads, num_key_value_heads, 
                 intermediate_size, token_size, rope_theta, name="qwen_cross_decoder", rope_max_len=4096):
        super().__init__(name)

        # RMSNorm Config
        self.input_rms_norm = RMSNorm(hidden_dim, 1e-8, name='input_rms_norm')
        self.ffn_input_rms_norm = RMSNorm(hidden_dim, 1e-8, name='ffn_input_rms_norm')

        # ROPE config'
        self.rope_max_len = rope_max_len
        
        self.register_buffer("cos_rop_max", None, persistent=False)
        self.register_buffer("sin_rop_max", None, persistent=False)


        # Attn Config
        self.hidden_dim = hidden_dim
        self.num_query_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.rope_theta = rope_theta

        self.dim = hidden_dim // num_attention_heads # 128 // 8 = 16
        self.kv_group_nums = num_attention_heads // num_key_value_heads # 8 // 2 = 4
        
        self.q_proj = nn.Linear(hidden_dim, num_attention_heads * self.dim)
        self.k_proj = nn.Linear(hidden_dim, num_key_value_heads * self.dim)
        self.v_proj = nn.Linear(hidden_dim, num_key_value_heads * self.dim)
        self.o_proj = nn.Linear(hidden_dim, num_attention_heads * self.dim)

        self._init_rope_rop_emb(self.rope_max_len, self.dim)

        # FFN config
        # SwishGLU

        self.gate_proj = nn.Linear(hidden_dim, intermediate_size)
        self.up_proj = nn.Linear(intermediate_size, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, intermediate_size)



        '''GQA:
            比如16个Q heads 2个 KV头, 8个Q就对应一个KV, 那么这个组数就是8
            在算的时候复制一下, 把KV复制8次; 相当于这8个Q对应一个KV

            标准做法（相邻分组）：
                假设 tensor 形状是 [Batch, Heads=4, Dim]。
                $Q_0$ 和 $Q_1$ 在物理内存里是挨在一起的。
                当我们想把 $Q_0, Q_1$ 拿出来和 $K_0$ 做计算时，我们可以直接用切片 Q[:, 0:2, :]。
                这在内存中是一次性的、连续的读取操作。非常快。
                操作代码：q.view(batch, 2, 2, dim) —— 这是一个 zero-copy（零拷贝）操作，只是改变了视图。

            你的做法（交错分组）：
                $Q_0$ 和 $Q_2$ 在内存里是分开的，中间隔着 $Q_1$。
                如果我们想把 $Q_0, Q_2$ 拿出来和 $K_0$ 做计算，我们需要用 stride（步长）切片 Q[:, 0::2, :]。
            这会导致内存访问不连续。在 GPU 上，这种非连续的 tensor 通常无法直接高效计算，必须先调用 .contiguous()
            这会触发一次内存拷贝（Memory Copy），把 $Q_0$ 和 $Q_2$ 搬到一个新的连续内存块里。
            代价：额外的显存占用 + 额外的时间开销。
                
            
            现在的 LLM 基本都依赖 FlashAttention 或 vLLM (PagedAttention) 来加速推理。
                这些底层 CUDA 核函数（Kernel）在设计时，都假设了一个前提：属于同一个 KV 头的 Q Heads 是连续存放的。
                FlashAttention 内部会把 $Q$ 当作一个块读进来。如果 $Q_0$ 和 $Q_1$ 共享同一个 $K$，Kernel 只需要加载一次 $K$，然后让 $Q_0, Q_1$ 依次计算。
        ''' 
    
    def _init_rope_rop_emb(self, seq_len, d_model):

        angle = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model)
        
        # [max_seq]
        max_seq = torch.arange(0, seq_len).unsqueeze(-1)

        # [max_seq, dim//2] -> [max_seq, dim] -> [1, 1, max_seq, dim]
        angle_with_seq = torch.matmul(max_seq, angle.unsqueeze(0)).repeat_interleave(2, dim=-1).unsqueeze(0).squeeze(0)

        
        self.cos_rop_max = angle_with_seq.cos()
        self.sin_rop_max = angle_with_seq.sin()

    def get_rope_emb(self, input:Tensor):
        # input [B, seq, dim]
        batch_size, real_encoder_seq, _ = input.shape

        if real_encoder_seq > self.rope_max_len:
            self.rope_max_len = real_encoder_seq
            self._init_rope_rop_emb(self.rope_max_len, self.dim)

        # [1, 1, max_len, dim]
        # self.cos_rop_max = angle_with_seq.cos()

        return self.cos_rop_max[:, :, 0:real_encoder_seq, :], self.sin_rop_max[:, :, 0:real_encoder_seq, :]



    def apply_rope(self, query:Tensor, key:Tensor, cos_rop:Tensor, sin_rop:Tensor):
        '''
        query [B, 8, seq, dim]
        key [B, 2, seq, dim]
        cos_rop: [1, 1, seq, dim]
        sin_rop: [1, 1, seq, dim]
        '''
        # 对于query key进行两两反转
        batchsize, query_head_num, seq_len, d_model = query.shape
        batchsize, key_head_num, seq_len, d_model = key.shape

        def trans(x:Tensor, head_nums):
            x = x.reshape(batchsize, head_nums, seq_len, d_model // 2, 2)
            # [B, head seq, d//2, 1]
            x1 = x[:, :, :, :, 0]
            x2 = -x[:, :, :, :, 1]
            # [B, seq, d//2, 2]
            # flatten交错填充
            return torch.stack( (x2, x1), dim=-1).reshape([batchsize, head_nums, seq_len,-1])

        query_trans = trans(query, query_head_num)
        key_trans = trans(key, key_head_num)

        # 广播乘法
        return cos_rop * query + sin_rop * query_trans, cos_rop * key + sin_rop * key_trans


    def gen_mask(self, seq_mask:Tensor) -> Tensor:
        # [B, seq, 1] -> [B, 8, seq, seq]
        return torch.matmul(seq_mask, seq_mask.transpose(1, 2)).unsqueeze(1).expand(-1, self.num_query_heads, -1, -1)


    # TODO 还差一个 casual mask

    def Qwen2Attn(self, cosin:Tensor, sin:Tensor, encoder_input:Tensor, seq_mask=None) -> Tensor:
        """
        encoder_input: [Batch, Seq, Dim]
        seq_mask: [Batch, Seq, 1] (Eecoder Mask) 理论上可以不用mask, 因为默认全部self-attn,但是可以更灵活控制需要看到哪些(万一有需要)

        seq_mask -> [B, seq, 1] 计算一个全1的矩阵即可 [seq, seq] 这样attn_scores就正常softmax
        """
        BatchSize, seq = encoder_input.shape[0], encoder_input.shape[1]

        # [B, seq, 8*dim]
        query:Tensor = self.q_proj(encoder_input)
        # [B, seq, 2*dim]
        key:Tensor = self.k_proj(encoder_input)
        # [B, seq, 2*dim]
        value:Tensor = self.v_proj(encoder_input)


        # [B, seq, 8, dim] -> [B, 8, seq, dim]
        query:Tensor = query.reshape([BatchSize, seq, self.num_query_heads, self.dim]).transpose(1, 2)
        # [B, seq, 2, dim] -> [B, 2, seq, dim]
        key:Tensor = key.reshape([BatchSize, seq, self.num_kv_heads, self.dim]).transpose(1, 2)
        # [B, seq, 2, dim] -> [B, 2, seq, dim]
        value:Tensor = value.reshape([BatchSize, seq, self.num_kv_heads, self.dim]).transpose(1, 2)

        query, key = self.apply_rope(query, key, cosin, sin)

        
        # 将key value对齐 [B, 2, seq, dim] -> [B, 8, seq dim]  [K0, K1]  ->  [K0, K0, K0, K0, K1, K1, K1, K1]
        # key = key.repeat_interleave(self.kv_group_nums, dim=1)
        # value = value.repeat_interleave(self.kv_group_nums, dim=1)


        # [[K0], [K1]] -> [[K0, K0, K0. K0], [K1, ..] ]
        key = key.unsqueeze(2).expand(-1, -1, self.kv_group_nums, -1, -1).reshape([BatchSize, -1, seq, self.dim])
        value = value.unsqueeze(2).expand(-1, -1, self.kv_group_nums, -1, -1).reshape([BatchSize, -1, seq, self.dim])


        # [B, 8, seq, seq]
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.dim)

        self_attn_seq_mask = self.gen_mask(seq_mask)

        # masked_fill_ 传入一个bool Tensor
        attn_scores = attn_scores.masked_fill_(self_attn_seq_mask == 0, -torch.inf)

        # [B, 8, seq, seq]
        attn_scores = torch.softmax(attn_scores, dim=-1)

        # [B, 8, seq, dim] -> [B, seq, 8*dim]

        ''' transpose 会修改步长, 使得逻辑上能够转置, 但是内存中是不连续的(不变), 如果调用view就会错, 但是reshape会保持深拷贝(改变内存中的结构，万能结构)'''
        attn_output = torch.matmul(attn_scores, value).transpose(1, 2).contiguous().reshape([BatchSize, seq, -1])

        # [B, seq, 8*dim] = [B, seq, hidden_dim]
        return self.o_proj(attn_output)

    def Qwen2FFN(self, ffn_input:Tensor):
        # FFN input : [B, seq, hidden_dim]

        # [B, seq, middle_dim]
        gate = self.gate_proj(ffn_input)

        ffn_up_layer = self.up_proj(ffn_input)

        # SWish
        gate = gate * torch.sigmoid(gate)

        ffn_up_layer = ffn_up_layer * gate

        ffn_down_layer = self.down_proj(ffn_up_layer)

        return ffn_down_layer


    def forward(self, encoder_input:Tensor, seq_mask=None) -> Tensor:
        '''
        PreNorm要保持最开始的输入, 而不是加norm后的输入
        '''
        encoder_input_normed = self.input_rms_norm(encoder_input)
        
        # ROPE
        cos_rop, sin_rop = self.get_rope_emb(encoder_input)
        # ATTN
        attn_output = encoder_input + self.Qwen2Attn(cos_rop, sin_rop, encoder_input_normed, seq_mask)

        ffn_input = attn_output
        ffn_input_norm = self.ffn_input_rms_norm(ffn_input)

        ffn_output = ffn_input + self.Qwen2FFN(ffn_input_norm)

        return ffn_output






class Qwen2DecoderCrossLayer(BaseNetwork):
    def __init__(self, hidden_dim, num_attention_heads, num_key_value_heads, 
                 intermediate_size, token_size, rope_theta, name="qwen_cross_decoder", rope_max_len=64):
        super().__init__(name)
        self.hidden_size = hidden_dim
        self.num_attention_heads = num_attention_heads
        

        # RMSNorm Config
        self.input_rms_norm = RMSNorm(hidden_dim, 1e-8, name='input_rms_norm')
        self.cross_attn_rms_norm = RMSNorm(hidden_dim, 1e-8, name='cross_attn_rms_norm')
        self.ffn_input_rms_norm = RMSNorm(hidden_dim, 1e-8, name='ffn_input_rms_norm')

        # ROPE config
        self.rope_max_len = rope_max_len
        self.cos_rop_max, self.sin_rop_max = None
        

        # Casual Attn Config
        self.hidden_dim = hidden_dim
        self.num_query_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.rope_theta = rope_theta

        self.dim = hidden_dim // num_attention_heads # 128 // 8 = 16
        self.kv_group_nums = num_attention_heads // num_key_value_heads # 8 // 2 = 4

        self.q_proj = nn.Linear(hidden_dim, num_attention_heads * self.dim)
        self.k_proj = nn.Linear(hidden_dim, num_key_value_heads * self.dim)
        self.v_proj = nn.Linear(hidden_dim, num_key_value_heads * self.dim)
        self.o_proj = nn.Linear(hidden_dim, num_attention_heads * self.dim)


        # Cross Attn Config

        self.q_proj_cross = nn.Linear(hidden_dim, num_attention_heads * self.dim)
        self.k_proj_cross = nn.Linear(hidden_dim, num_key_value_heads * self.dim)
        self.v_proj_cross = nn.Linear(hidden_dim, num_key_value_heads * self.dim)
        self.o_proj_cross = nn.Linear(hidden_dim, num_attention_heads * self.dim)

        



        # FFN config
        # SwishGLU
        self.gate_proj = nn.Linear(hidden_dim, intermediate_size)
        self.up_proj = nn.Linear(intermediate_size, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, intermediate_size)





    def forward(self, decoder_input, encoder_fusion, seq_mask=None, kv_mask=None):
        """
        x: [Batch, seq_decoder, Dim]
        memory: [Batch, Seq_len_src, Dim] (Encoder Output)
        seq_mask: [Batch, seq_decoder, 1] (Decoder Mask)
        kv_mask: [Batch, seq_encoder, 1] (Encoder Mask)
        """

        # [B, seq_d, dim]
        decoder_input_normed = self.input_rms_norm(decoder_input)

        cos_rop, sin_rop = self.get_rope_emb(decoder_input) 

        # [B, seq_d, dim]
        decoder_casual_attn_output = decoder_input + self.qwen2CasualAttn(cos_rop, sin_rop, decoder_input_normed, seq_mask)
        
        decoder_cross_attn_input_normed = self.cross_attn_rms_norm(decoder_casual_attn_output)

        decoder_cross_attn_output = decoder_casual_attn_output + self.qwen2CrossAttn(decoder_cross_attn_input_normed, encoder_fusion, seq_mask, kv_mask)
        
        decoder_ffn_input_normed = self.ffn_input_rms_norm(decoder_cross_attn_output)

        decoder_ffn_output = decoder_cross_attn_output + self.qwen2FFN(decoder_ffn_input_normed)

        return decoder_ffn_output

    
    def _init_rope_rop_emb(self, max_seq_len, d_model):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        angle = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model)
        
        # [max_seq]
        max_seq = torch.arange(0, max_seq_len).unsqueeze(-1)

        # [max_seq, dim//2] -> [max_seq, dim] -> [1, 1, max_seq, dim]
        angle_with_seq = torch.matmul(max_seq, angle.unsqueeze(0)).repeat_interleave(2, dim=-1).unsqueeze(0).squeeze(0)

        self.cos_rop_max = angle_with_seq.cos()
        self.sin_rop_max = angle_with_seq.sin()

    def get_rope_emb(self, input:Tensor):
        # input [B, seq, dim]
        batch_size, real_decoder_seq, _ = input.shape

        # [1, 1, max_len, dim]
        # self.cos_rop_max = angle_with_seq.cos()

        return self.cos_rop_max[:, :, 0:real_decoder_seq, :], self.sin_rop_max[:, :, 0:real_decoder_seq, :]



    def apply_rope(self, query:Tensor, key:Tensor, cos_rop:Tensor, sin_rop:Tensor):
        '''
        query [B, 8, seq, dim]
        key [B, 2, seq, dim]
        cos_rop: [1, 1, seq, dim]
        sin_rop: [1, 1, seq, dim]
        '''
        # 对于query key进行两两反转
        batchsize, query_head_nums, seq_len, d_model = query.shape
        batchsize, key_head_nums, seq_len, d_model = key.shape

        def trans(x:Tensor, head_nums):
            x = x.reshape(batchsize, head_nums, seq_len, d_model // 2, 2)
            # [B, head, seq, d//2, 1]
            x1 = x[:, :, :, :, 0]
            x2 = -x[:, :, :, :, 1]

            # [B, head, seq, d//2, 2] -> [B, head, seq, d]
            # flatten交错填充
            return torch.stack( (x2, x1), dim=-1).reshape([batchsize, head_nums, seq_len, -1])

        query_trans = trans(query, query_head_nums)

        key_trans = trans(key, key_head_nums)



        return cos_rop * query + sin_rop * query_trans, cos_rop * key + sin_rop * key_trans


    def gen_casual_mask(self, seq_mask:Tensor):
        # seq_mask [B, seq_decoder]  [1 1 1 0 0 ], [1, 1, 1, 1, 0]

        BatchSize, seq = seq_mask.shape

        # [seq, seq]

        # TODO: tril用法

        mask = torch.tril(torch.ones(seq, seq, device=seq_mask.device))  # 下面是用的mask_fill_(bool_tensor)

        # [B, seq, seq] 
        padding_mask = torch.matmul(seq_mask.unsqueeze(-1), seq_mask.unsqueeze(-1).transpose(-1, -2))
        mask = mask * padding_mask

        # MAKE 下三角矩阵 [B, 1, seq, seq]  8 = nums_of_query_heads
        return mask.unsqueeze(1)

    def qwen2CasualAttn(self, cos_rop:Tensor, sin_rop:Tensor, decoder_input:Tensor, seq_mask):
        BatchSize, seq, dim = decoder_input.shape

        # [B, seq, 8*dim]
        query_init:Tensor = self.q_proj(decoder_input)
        # [B, seq, 2*dim]
        key_init:Tensor = self.k_proj(decoder_input)
        # [B, seq, 2*dim]
        value:Tensor = self.v_proj(decoder_input)


        # [B, seq, 8, dim] -> [B, 8, seq, dim]
        query:Tensor = query_init.reshape([BatchSize, seq, self.num_query_heads, self.dim]).transpose(1, 2)
        # [B, seq, 2, dim] -> [B, 2, seq, dim]
        key:Tensor = key_init.reshape([BatchSize, seq, self.num_kv_heads, self.dim]).transpose(1, 2)
        # [B, seq, 2, dim] -> [B, 2, seq, dim]
        value:Tensor = value.reshape([BatchSize, seq, self.num_kv_heads, self.dim]).transpose(1, 2)

        query , key = self.apply_rope(query, key, cos_rop, sin_rop)


        
        # 将key value对齐 [B, 2, seq, dim] -> [B, 8, seq dim]  [K0, K1]  ->  [K0, K0, K0, K0, K1, K1, K1, K1]
        # key = key.repeat_interleave(self.kv_group_nums, dim=1)
        # value = value.repeat_interleave(self.kv_group_nums, dim=1)

        # [[K0], [K1]] -> [[K0, K0, K0. K0], [K1, ..] ]
        key = key.unsqueeze(2).expand(-1, -1, self.kv_group_nums, -1, -1).reshape([BatchSize, -1, seq, self.dim])
        value = value.unsqueeze(2).expand(-1, -1, self.kv_group_nums, -1, -1).reshape([BatchSize, -1, seq, self.dim])


        # [B, 8, seq, seq]
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.dim)

        # [B, 1, seq, seq]
        casual_attn_seq_mask = self.gen_casual_mask(seq_mask)

        # masked_fill_ 传入一个bool Tensor
        attn_scores = attn_scores.masked_fill_(casual_attn_seq_mask == 0, -torch.inf)

        # [B, 8, seq, seq]
        attn_scores = torch.softmax(attn_scores, dim=-1)

        # [B, 8, seq, dim] -> [B, seq, 8*dim]

        ''' transpose 会修改步长, 使得逻辑上能够转置, 但是内存中是不连续的(不变), 如果调用view就会错, 但是reshape会保持深拷贝(改变内存中的结构)'''
        attn_output = torch.matmul(attn_scores, value).transpose(1, 2).contiguous().reshape([BatchSize, seq, -1])

        # [B, seq, 8*dim] = [B, seq, hidden_dim]
        return self.o_proj(attn_output)

    def gen_kv_mask(self, seq_mask:Tensor, kv_mask:Tensor):
        # seq_mask [B, seq_decoder]  [1 1 1 0 0 0]
        # kv_mask [B, seq_encoder]   [1 1 0 0]

        Batchsize = seq_mask.shape[0]
        # [B, seq_d, seq_e] -> [B, 8, seq_d, seq_e]
        return torch.matmul(seq_mask.unsqueeze(-1), kv_mask.unsqueeze(-1).transpose(-1, -2)).unsqueeze(1).expand(Batchsize, self.num_query_heads, -1, -1)



    def qwen2CrossAttn(self, decoder_input:Tensor, encoder_fusion:Tensor, seq_mask:Tensor, kv_encoder_mask:Tensor):
        """
        
        :param decoder_input:  [B, seq_d, hidden_dim]
        :param encoder_fusion: [B, seq_e, hidden_dim]
        :param seq_mask:  [B, seq_d, 1]
        :param kv_encoder_mask: [B, seq_e, 1]
        """

        BatchSize, seq_decoder, seq_encoder = decoder_input.shape[0], decoder_input.shape[1], encoder_fusion.shape[1]

        # [B, seq_d, 8*dim]
        query:Tensor = self.q_proj_cross(decoder_input)
        # [B, seq_e, 2*dim]
        key:Tensor = self.k_proj_cross(encoder_fusion)
        # [B, seq_e, 2*dim]
        value:Tensor = self.v_proj_cross(encoder_fusion)


        # [B, seq_d, 8, dim] -> [B, 8, seq_d, dim]
        query:Tensor = query.reshape([BatchSize, seq_decoder, self.num_query_heads, self.dim]).transpose(1, 2)
        # [B, seq_e, 2, dim] -> [B, 2, seq_e, dim]
        key:Tensor = key.reshape([BatchSize, seq_encoder, self.num_kv_heads, self.dim]).transpose(1, 2)
        # [B, seq_e, 2, dim] -> [B, 2, seq_e, dim]
        value:Tensor = value.reshape([BatchSize, seq_encoder, self.num_kv_heads, self.dim]).transpose(1, 2)

        
        # 将key value对齐 [B, 2, seq_e, dim] -> [B, 8, seq_e dim]  [K0, K1]  ->  [K0, K0, K0, K0, K1, K1, K1, K1]
        # key = key.repeat_interleave(self.kv_group_nums, dim=1)
        # value = value.repeat_interleave(self.kv_group_nums, dim=1)

        # [[K0], [K1]] -> [[K0, K0, K0. K0], [K1, ..] ]
        key = key.unsqueeze(2).expand(-1, -1, self.kv_group_nums, -1, -1).reshape([BatchSize, -1, seq_encoder, self.dim])
        value = value.unsqueeze(2).expand(-1, -1, self.kv_group_nums, -1, -1).reshape([BatchSize, -1, seq_encoder, self.dim])

        # key [B, 8, seq_e, dim] 

        # [B, 8, seq_d, seq_e]
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.dim)
        # [B, 8, seq_d, seq_e]
        casual_attn_seq_mask = self.gen_kv_mask(seq_mask, kv_encoder_mask)

        # masked_fill_ 传入一个bool Tensor
        attn_scores = attn_scores.masked_fill_(casual_attn_seq_mask == 0, -torch.inf)

        # [B, 8, seq_d, seq_e]
        attn_scores = torch.softmax(attn_scores, dim=-1)

        # [B, 8, seq_d, seq_e] @ [B, 8, seq_e, dim] = [B, 8, seq_d, dim] -> [B, seq_d, 8*dim]

        ''' transpose 会修改步长, 使得逻辑上能够转置, 但是内存中是不连续的(不变), 如果调用view就会错, 但是reshape会保持深拷贝(改变内存中的结构)'''
        attn_output = torch.matmul(attn_scores, value).transpose(1, 2).contiguous().reshape([BatchSize, seq_decoder, -1])

        # [B, seq, 8*dim] = [B, seq, hidden_dim]
        return self.o_proj_cross(attn_output)

        
    def qwen2FFN(self, ffn_input:Tensor):  
        # FFN input : [B, seq, hidden_dim]

        # [B, seq, middle_dim]
        gate = self.gate_proj(ffn_input)

        ffn_up_layer = self.up_proj(ffn_input)

        # SWish
        gate = gate * torch.sigmoid(gate)

        ffn_up_layer = ffn_up_layer * gate

        ffn_down_layer = self.down_proj(ffn_up_layer)

        return ffn_down_layer