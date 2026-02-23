import torch
import torch.nn as nn
from torch import Tensor

class Generator(nn.Module):
    def __init__(self, init_dim):
        super().__init__()
        # [B, 1, ]

        self.linear = nn.Linear(init_dim, 64*4*4)

        # [B, 64, 4, 4] -> [B, 64, 8, 8]
        self.down1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.down2 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.down3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.down4 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.down5 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.down6 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
        # [B, 3, 256, 256]

        self.batchnorm0 = nn.BatchNorm2d(64)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(64)


        self.leakey = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # [B, init_dim] -> [B, 256, 256]
        batchsize = x.shape[0]
        x = self.linear(x).view(batchsize, 64, 4, 4)
        x = self.leakey(x)

        out = x

        out = self.down1(out)
        out = self.batchnorm0(out)
        out = self.leakey(out)

        out = self.down2(out)
        out = self.batchnorm1(out)
        out = self.leakey(out)

        out = self.down3(out)
        out = self.batchnorm2(out)
        out = self.leakey(out)

        out = self.down4(out)
        out = self.batchnorm3(out)
        out = self.leakey(out)

        out = self.down5(out)
        out = self.batchnorm4(out)
        out = self.leakey(out)

        out = self.down6(out)

        out = self.tanh(out)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, ):
        super().__init__()
        # [3, 256, 256]

        # [64, 128, 128]
        self.up1 = nn.Conv2d(3, 64, 3, 2, 1)  
        
        # [64, 64, 64]
        self.up2 = nn.Conv2d(64, 64, 3, 2, 1)
        # [64, 32, 32]
        self.up3 = nn.Conv2d(64, 64, 3, 2, 1)
        # [64, 16, 16]
        self.up4 = nn.Conv2d(64, 64, 3, 2, 1)

        self.up5 = nn.Conv2d(64, 64, 3, 2, 1)

        self.linear1 = nn.Linear(64*8*8, 64)
        self.linear2 = nn.Linear(64,1)

        self.leakey = nn.LeakyReLU(0.2)

        self.sigmoid = nn.Sigmoid()

        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(64)



    def forward(self, x):
        # [B, 3, 256, 256] -> [B, 1]
        batchsize = x.shape[0]
        out = self.up1(x)
        out = self.leakey(out)

        out = self.up2(out)
        out = self.batchnorm1(out)
        out = self.leakey(out)

        out = self.up3(out)
        out = self.batchnorm2(out)
        out = self.leakey(out)

        out = self.up4(out)
        out = self.batchnorm3(out)
        out = self.leakey(out)

        out = self.up5(out)
        out = self.batchnorm4(out)
        out = self.leakey(out)

        out = self.linear1(out.view(batchsize, -1))
        out = self.leakey(out)
        out = self.linear2(out)
        out = self.sigmoid(out)

        return out #[B, 1]