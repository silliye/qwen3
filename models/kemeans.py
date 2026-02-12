
import torch
from torch import Tensor

from torch.optim import AdamW

from torch.utils.data import DataLoader, Dataset


class Kmeans():
    # 批量更新
    def __init__(self, k: int, emb_dim,  mini_inter=1):
        self.k = k
        self.emb_dim = emb_dim
        self.mini_inter = mini_inter

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # init
        self.clusters = None # [k, emb]
        self.if_init_clusters = False

    def _init_clusters(self, X:Tensor):
        # X [batch, emb]
        batch, dim = X.shape
        self.clusters = X[torch.randperm(batch, device=X.device)[:self.k], :].clone()
        
    @torch.no_grad()
    def fit(self, X: Tensor):

        if not self.if_init_clusters:
            self.if_init_clusters = True
            self._init_clusters(X)

        # X [batch, emb_dim], clusters [k, emb_dim]
        batchsize, dim = X.shape

        # [batch, k]
        dis = torch.cdist(X, self.clusters)
        # [bacth,]
        indices = dis.argmin(-1)

        new_clusters = self.clusters.clone()
        for i in range(self.k):

            mask = (indices == i)
            if mask.any():
                new_clusters[i] = X[mask].mean(dim=0, keepdim=False)
            
        self.clusters = new_clusters
        
    @torch.no_grad()
    def predict(self, X: Tensor):

        # [B, K]
        dis = torch.cdist(X, self.clusters)
        return dis.argmin(-1)


class EmbeddingDataSet(Dataset):    
    def __init__(self, embedding:Tensor):
        super().__init__()
        self.embs = embedding
    
    def __len__(self):
        return self.embs.shape[0]

    def __getitem__(self, index):
        return self.embs[index]

def train():
    embedding = torch.randn(500, 10)
    embedding_dataset = EmbeddingDataSet(embedding)
    embedding_dataloader = DataLoader(embedding_dataset, batch_size=50, shuffle=True)

    model = Kmeans(k=5, emb_dim=10)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    epoches = 2
    for epoch in range(epoches):
        for batch in embedding_dataloader:
            batch = batch.to(device)

            model.fit(batch)
    

    for batch in embedding_dataloader:
            batch.to(device)

            indices = model.predict(batch)
            print(indices)


# train()

x = Tensor([[3, 3, 3], [3, 2, 1]])
print(x[0])
mask = Tensor([0]).int
x[mask] = x[mask] - Tensor([2, 2, 2])

print(x)
a = Tensor([1, 7, 5, 7])
b = Tensor([2, 5, 6, 1])
d = Tensor([33, 55, 66, 11])


c = torch.stack([a, b, d], dim=0)
print(c)