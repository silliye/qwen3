import torch
from torch import Tensor


class RQKmeans():


    def __init__(self, res_layers, k, emb_dim):
        self.res_layers = res_layers
        self.k = k
        self.emb_dim = emb_dim

        self.if_init_cluster = False

        self.clusters_layers = None

    
    def _init_cluster(self, X:Tensor):
        batchsize, _ = X.shape

        # [3, K, dim]
        self.clusters_layers = X[torch.randperm(batchsize)[:self.k]].clone().unsqueeze(0).expand(self.res_layers, -1, -1).clone()

        

    @torch.no_grad()
    def fit(self, X:Tensor):
        if not self.if_init_cluster:
            self.if_init_cluster = True
            self._init_cluster(X)

        # X[B, dim]  cluster[3, K, dim]
        batchsize, _ = X.shape
        clusters_layers_copy = self.clusters_layers.clone()

        residual = X.clone()
        for layer_index in range(self.res_layers):
            
            residual_copy = residual.clone()

            # [B, K]
            dis = torch.cdist(residual, self.clusters_layers[layer_index])
            indice = dis.argmin(dim=-1)

            for k_index in range(self.k):
                hit = (indice == k_index)

                if hit.any():
                    center = residual[hit].mean(0)
                    clusters_layers_copy[layer_index, k_index] = center

                    # broadcast [B]
                    residual_copy[hit] = residual[hit] - center

            residual = residual_copy
        
        self.clusters_layers = clusters_layers_copy

    @torch.no_grad()
    def predict(self, X:Tensor):
        indices = []
        residual = X.clone()
        for layer in range(self.res_layers):
            
            indice = torch.cdist(residual, self.clusters_layers[layer, :, :]).argmin(-1)

            residual = residual - self.clusters_layers[layer][indice]

            indices.append(indice)


        return torch.stack(indices, dim=-1)            