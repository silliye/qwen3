import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.nn import all_gather



def infonceloss(X, Y, tau):
    batch_size, dim = X.shape
    # [single_batchsize, dim]
    X, Y = F.normalize(X, dim=-1), F.normalize(Y, dim=-1)

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1
    
    # [bs, dim] -> [world_size, bs, dim]
   
    X_all = all_gather(X)
    Y_all = all_gather(Y)

    X_all = torch.cat(X_all, dim=-1)
    Y_all = torch.cat(Y_all, dim=-1)
    
    # [bs, dim] * [dim, bs*ws] -> [bs, bs*ws]
    sim1 = X @ Y_all.T / tau
    sim2 = Y @ X_all.T / tau

    # [bs]
    labels = torch.arange(batch_size, device=X.device) + rank * batch_size

    return (F.cross_entropy(sim1, labels) + F.cross_entropy(sim2, labels)) / 2

    

    



