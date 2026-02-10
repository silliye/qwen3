import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils import MLP 

from torch import Tensor

class RQVAEModel(BaseModel):
    def __init__(self, config:dict):
        super(RQVAEModel, self).__init__()
        
        self.codebook_size = config.get('codebook_size', 301)
        self.rq_level = int(config.get('rq_level', 2))
        self.experts_num = int(config.get('experts_num'), 4)
        self.code_dim = int(config.get('code_dim', 16))
        self.hidden_dim = int(config.get('hidden_dim', 64))
        self.input_quan_dim = int(config.get('input_quan_dim', 64))
        self.rebuild_loss_weight = float(config.get("rebuild_loss_weight", 1.0))
        self.align_weight = float(config.get("align_weight", 1.0))
        self.align_weight_beta = float(config.get("align_weight_beta", 1.0))

        
        

        self.experts = nn.ModuleList([
            MLP(input_dim=self.input_quan_dim, hidden_dims=[self.hidden_dim], output_dim=self.code_dim) for _ in range(self.experts_num)
        ])        

        self.router = MLP(input_dim=self.code_dim, hidden_dims=[], output_dim=self.experts_num, activation=nn.Sigmoid, if_act_last=True)

        self.decoder = MLP(
            input_dim=self.code_dim * self.experts_num,
            hidden_dims=[self.hidden_dim],
            output_dim=self.input_quan_dim,
            name="decoder_mlp"
        )

        # 用 ModuleList 管理多层码本，类似 TF 的 variable scope 概念，但这里是 list
        self.codebooks = nn.ModuleList([
            nn.Embedding(self.codebook_size, self.code_dim) for _ in range(self.experts_num * self.rq_level)
        ])
        
        # 初始化权重 TF 的 truncated_normal stddev=0.36)
        for cb in self.codebooks:
            nn.init.trunc_normal_(cb.weight, std=0.36)

    def forward(self, input_data):
        """
        input_data: Tensor [Batch, input_quan_dim]
        """
        # 假设 input_data 已经是 float tensor
        input_emb = input_data / 1e6 
        
        vq_loss, code_idxs = self.auto_opmq(input_emb)
        
        return vq_loss, code_idxs

    def auto_opmq(self, input_embedding):
        code_emb_list = []
        code_index_list = []
        hidden_emb_list = []

        align_loss_1 = 0
        align_loss_2 = 0

        # 直接调用 self.encoder 对象

        for idx in range(self.experts_num):
            
            hidden_emb:Tensor = self.experts[idx](input_embedding)
            
            # 取出当前层的码本
            codebook_layer = self.codebooks[idx]


            # 搜索最近邻
            # quan_emb: [B, dim]
            # code_emb: [B, dim], code_index: [B, 1]
            code_emb, code_index = self.search_codebook(hidden_emb, codebook_layer)


            # 计算对齐损失 (Align Loss)
            # .detach() 等同于 tf.stop_gradient()
            
            # sg(ze) - e -> 让 e 靠近 ze
            loss_1 = (hidden_emb.detach() - code_emb).pow(2).mean()
            # sg(e) - ze -> 让 ze 靠近 e
            loss_2 = (code_emb.detach() - hidden_emb).pow(2).mean()

            align_loss_1 += loss_1
            align_loss_2 += loss_2

            code_index_list.append(code_index)

            hidden_emb_list.append(hidden_emb)


            # 更新残差: 下一层的输入 = 当前输入 - 当前量化向量
            hidden_diff = hidden_emb - code_emb

            codebook_res_layer = self.codebooks[idx+self.experts_num]
            
            code_diff_emb, code_res_index = self.search_codebook(hidden_diff, codebook_res_layer)

             # sg(ze) - e -> 让 e 靠近 ze
            loss_1 = (hidden_diff.detach() - code_diff_emb).pow(2).mean()
            # sg(e) - ze -> 让 ze 靠近 e
            loss_2 = (code_diff_emb.detach() - hidden_diff).pow(2).mean()

            align_loss_1 += loss_1
            align_loss_2 += loss_2

            code_index_list.append(code_res_index)


            code_emb_list.append(code_emb + code_diff_emb)
            

        # --- 3. Rebuild (Decode) ---
        # 将所有层的 code_emb 相加恢复
        # stack list -> list of  [B, dim] -> [B, dim*4]
        all_codes = torch.concat(code_emb_list, dim=-1)
        all_hidden_embs = torch.concat(hidden_emb_list, dim=-1)
        
        # 梯度直通估计 (Straight-Through Estimator trick)
        # 前向传播用 all_codes 的值，反向传播梯度传给 encoder_quan_emb
        decoder_input = all_hidden_embs + (all_codes - all_hidden_embs).detach()
        
        # Decode
        rebuild_emb = self.decoder(decoder_input)

        # Rebuild Loss
        rebuild_loss = (rebuild_emb - input_embedding).pow(2).mean()

        # 平均 align loss
        align_loss_1 = align_loss_1 / (self.rq_level * self.experts_num)
        align_loss_2 = align_loss_2 / (self.rq_level * self.experts_num)

        # 总 Loss
        vq_loss = rebuild_loss * self.rebuild_loss_weight + \
                  (align_loss_1 + align_loss_2 * self.align_weight_beta) * self.align_weight
        
        self.losses.append(vq_loss.item()) # 记录数值用于打印

        return vq_loss, code_index_list

    def search_codebook(self, inputs, codebook_layer):
        """
        inputs: [B, dim]
        codebook_layer: nn.Embedding 对象
        """
        # [code_size, dim]
        embedding_weight = codebook_layer.weight
        
        # 计算距离矩阵 (利用广播机制)
        # inputs: [B, 1, dim]
        inputs_expanded = inputs.unsqueeze(1)
        # weights: [1, code_size, dim]
        weights_expanded = embedding_weight.unsqueeze(0)
        
        # dist: [B, code_size]
        # norm(dim=-1) 计算最后一维的 L2 范数
        dist = torch.norm(inputs_expanded - weights_expanded, p=2, dim=-1)
        
        # argmin 找最近的下标 [B]
        min_encoding_indices = torch.argmin(dist, dim=1)
        
        # 获取对应的向量 [B, dim]
        # nn.Embedding 可以直接用下标取向量
        quantized_out = codebook_layer(min_encoding_indices)
        
        return quantized_out, min_encoding_indices.unsqueeze(1) # 返回 [B, 1] 保持形状一致