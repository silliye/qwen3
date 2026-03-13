import torch
import torch.nn as nn
from torch import Tensor

class Dice(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.batchnorm = nn.BatchNorm1d(1)
    def pfunc(self, X):
        # [B]
        B = X.shape
        eposilon = 1e-5
        return 1 / (1 + torch.exp(-self.batchnorm(X.view(B, 1))))

    def forward(self, X):
        # [Batch, seq, 1]
        batch, seq, _ = X.shape
        X = X.view(-1)
        ps = self.pfunc(X)
        v = ps * X + (1-ps) * X * self.alpha
        return v.view(batch, seq, 1)
    
class AttentionUnit(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.action_func = Dice()
        self.linear = nn.Linear(dim*3, 1)

    def forward(self, X:Tensor, target:Tensor):
        # X.     [batch, seq, dim]
        # target [batch, 1, dim]
        # return [batch, seq, 1]
        
        # [batch, seq, dim]
        batch, seq, dim = X.shape
        mul = X * target
        return self.action_func(self.linear(torch.concat([X, target.expand(-1, seq, -1), mul], dim=-1)))

class DIN(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.attentionUnit = AttentionUnit(dim)

    def forward(self, X, target):
        # X[batch, seq, dim]
        # target [batch, dim]
        batch, seq, dim = X.shape
        # [batch, seq, 1]
        logits = self.attentionUnit(X, target.unsqueeze(1))

        return torch.sum(logits * X, dim=1, keepdim=False)



