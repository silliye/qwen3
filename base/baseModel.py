import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.losses = []
    
    def get_losses(self):
        return self.losses