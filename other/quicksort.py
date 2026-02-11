import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class LoraAdapter(nn.Module):
    def __init__(self, lora_rank, lora_alpha, linear:nn.Linear, lora_dropout=0.01):
        super().__init__()
        self.rank = lora_rank # 4
        self.alpha = lora_alpha # 8
        self.lora_dropout = lora_dropout
        self.W_A = nn.Parameter(torch.empty(linear.in_features, lora_rank))
        self.W_B = nn.Parameter(torch.empty(lora_rank, linear.out_features))
        
        self._weight_init_()
        # [in, out]
        self.init_linear = linear
        self.init_linear.weight.requires_grad = False
        self.dropout = nn.Dropout(lora_dropout)
        self.if_merge = False

    def _weight_init_(self):
        nn.init.kaiming_normal_(self.W_A)
        nn.init.zeros_(self.W_B)

    def reweight(self):
        self.init_linear.weight += (self.alpha / self.rank) * self.W_B.T @ self.W_A.T
        # init_linear.weight [out, in]
    
    def unmerge(self):
        self.init_linear.weight -= (self.alpha / self.rank) * self.W_B.T @ self.W_A.T

    def forward(self, x):
        
        if self.training:
            if self.if_merge:
                self.unmerge()
                self.if_merge = False
            return self.init_linear(x) + (self.alpha / self.rank) * self.dropout(x) @ self.W_A @ self.W_B
        else:
            if not self.if_merge:
                self.reweight()
                self.if_merge = True
            return self.init_linear(x)
