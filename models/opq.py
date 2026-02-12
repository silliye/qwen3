import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from utils import MLP 

from torch import Tensor

class OPQModel(BaseModel):
    def __init__(self, config:dict):
        super(OPQModel, self).__init__()
        
        self.codebook_size = config.get('codebook_size', 301)
        
        self.hidden_dim = int(config.get('hidden_dim', 64))
        self.input_quan_dim = int(config.get('input_quan_dim', 64))


        self.opq_heads = int(config.get('opq_heads', 8))
        self.code_dim = self.input_quan_dim // self.opq_heads
        

        self.rebuild_loss_weight = float(config.get("rebuild_loss_weight", 1.0))
        self.align_weight = float(config.get("align_weight", 1.0))
        self.align_weight_beta = float(config.get("align_weight_beta", 1.0))

        
        self.orth_encoder = nn.Linear(self.input_quan_dim, self.input_quan_dim)

        
        self.decoder = MLP(
            input_dim=self.code_dim * self.opq_heads,
            hidden_dims=[self.hidden_dim],
            output_dim=self.input_quan_dim,
            name="decoder_mlp"
        )

        # 用 ModuleList 管理多层码本，类似 TF 的 variable scope 概念，但这里是 list
        self.codebooks = nn.ModuleList([
            nn.Embedding(self.codebook_size, self.code_dim) for _ in range(self.opq_heads)
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
        
        input_embedding = input_embedding / 1e6
        # [B, 64]
        quan_emb:Tensor = self.orth_encoder(input_embedding)

        head_embs = quan_emb.split(self.opq_heads, dim=-1)

        code_emb_total = []
        auxloss = 0
        for i, emb in enumerate(head_embs):
            code_emb, indice = self.search_codebook(emb, self.codebooks[i])
            code_emb_total.append(code_emb)

            err1 = emb.detach() - code_emb
            err2 = emb - code_emb.detach()

            aux = err1.pow(2).mean() + self.align_weight_beta * err2.pow(2).mean()

            auxloss += aux

        # [B, dim*8]
        code_emb_concat = torch.concat(code_emb_total, dim=-1)

        rebuild_emb = self.decoder(head_embs + (code_emb_concat - head_embs).detach())
        
        rebuild_loss = torch.pow(rebuild_emb - quan_emb, 2).mean()

        self.losses += rebuild_loss * self.rebuild_loss_weight + auxloss * self.align_weight

        Weight = self.orth_encoder.weight # [64, 64]
        self.losses += (Weight @ Weight.T - torch.eye(self.input_quan_dim)).mean()
        # quan_emb  code_emb_total
        
        return 


    def search_codebook(self, inputs, codebook_layer):
        """
        inputs: [B, dim]
        codebook_layer: nn.Embedding 对象 [code_size, dim]
        """

        inputs = F.normalize(inputs, dim=-1)
        codebook_layer = F.normalize(codebook_layer.weight, dim=-1)

        # [B, ]
        indices = torch.cdist(inputs, codebook_layer).argmin(dim=-1)
        # [B, dim]
        return codebook_layer[indices], indices

       