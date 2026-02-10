'''
在这里写一些torch.api 熟悉一下
'''

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import Tensor



def sigmoid(x):
    # x [B,]
    return 1 / (1 + torch.exp(-x))

def softmax(x, tau, axis=-1):
    # x: [B, L]
    # [B, L]
    max_value = torch.max(x, axis=-1, keepdim=True).values
    x = x - max_value

    x = x / tau

    exp = torch.exp(x)
    exp_sum = torch.sum(exp, axis=-1, keepdim=True)

    return exp / exp_sum


def binary_cross_entropy(logits, labels):
    eposilon = 1e-8
    # [B, 1]
    # [B, 1]
    labels = labels.unsqueeze(-1)
    logits = sigmoid(logits)
    logits = torch.clip(logits, eposilon, 1.0-eposilon)

    return - torch.mean(labels * torch.log(logits) + (1-labels)*torch.log(1-logits), dim=-1)

def logsoftmax(x):
    # [B, L]
    # 算softmax的时候一定要注意数值稳定
    x = x - torch.max(x, dim=-1, keepdim=True).values
    return x - torch.log(torch.sum(torch.exp(x), dim=-1, keepdim=True))


def cross_entropy(logits, labels):
    # logits : [B, L]
    # lables : [B]
    logits = -logsoftmax(logits)

    # 从第二个维度开始取
    return torch.mean(torch.gather(logits, index=labels.view(-1, 1), dim=-1))

logits = torch.Tensor([[1, 1, 1], [2, 1, 0]])
lables = torch.LongTensor([1, 0])

print(cross_entropy(logits, lables))


def infonceloss(x1:Tensor, x2:Tensor, tau=0.07):
    batchsize, dim = x1.shape
    # x1 = [B, L]
    # x2 = [B, L]
    x1 = F.normalize(x1, p=2, dim=-1)
    x2 = F.normalize(x2, p=2, dim=-1)

    # [B, B]
    sim1 = x1 @ x2.T / tau
    
    # [B]
    labels = torch.arange(batchsize, device=x1.device)

    loss = cross_entropy(sim1, labels) + cross_entropy(sim1.T, labels)

    return loss / 2

x1 = torch.Tensor([[1, 1, 1], [2, 1, 0]])
x2 = torch.Tensor([[2, 3, 1], [6, -2, 0]])
print(infonceloss(x1, x2))


def auc_(label:list, logits:list) -> float:
    size = len(label)
    neg_samples = []
    pos_samples = []

    for i in range(size):
        if label[i] == 0: 
            neg_samples.append(logits[i])
        elif label[i] == 1: 
            pos_samples.append(logits[i])
        else:
            assert 1, 'except'
    
    neg_nums = len(neg_samples)
    pos_nums = len(pos_samples)
    hits = 0
    for i in range(pos_nums):
        for j in range(neg_nums):
            if pos_samples[i] > neg_samples[j]:
                hits += 1 
            elif pos_samples[i] == neg_samples[j]:
                hits += 0.5
    
    return hits / (neg_nums * pos_nums)

def auc(pos:Tensor, neg):
    # pos [B, ]
    # neg [B, ]

    size = pos.shape



    return 

def grad_descent():
    return 


def sqrt_2():

    # 二分查找
    # x^2 - 2
    l = 0
    r = 2
    mid = 1
    eposilon = 1e-6
    count = 100000
    while count and (mid ** 2 - 2)**2 > eposilon:
        mid = (l+r) / 2
        if mid ** 2 < 2:
            l = mid
        else:
            r = mid
        count -= 1
    return mid
x = sqrt_2()
# print(x)

import random 
import time
def pi():
    times = 10000000
    hits = 0
    t = time.time()
    for i in range(times):
    
        x = random.random()
        y = random.random()

        dis = x ** 2 + y ** 2
        if dis <= 1:
            hits += 1
    t2 = time.time()
    
    return hits / times * 4, t2-t

def pi2():
    B = 10000000
    t = time.time()
    random_ = torch.rand(B, 2)
    count = (random_[:, 0]**2 + random_[:, 1]**2 <= 1.0).sum().item()
    t2 = time.time()
    return count * 4 / B, t2-t

print(pi())
print(pi2())