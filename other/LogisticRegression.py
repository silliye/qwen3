# 手写一个逻辑回归

# 记住导数:
# y->z = 

import math
import random


class LogisticRegression:
    def __init__(self):
        self.weight1 = 0.1
        self.weight2 = 0.1
        self.bias = 0.0

        self.grad_weight1 = 0
        self.grad_weight2 = 0
        self.grad_bias = 0

        self.alpha = 0.04

        self.compute_grad = {'x1':0, 'x2':0, 'bias':0}
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    

    def sigmoid_grad(self, x):
        # return (math.exp(-x)) / (1+math.exp(-x)) ** 2
        return self.sigmoid(x)(1- self.sigmoid(x))


    def forward(self, x1, x2):
        self.compute_grad['x1'] = x1
        self.compute_grad['x2'] = x2
        self.compute_grad['bias'] = self.bias
        self.compute_grad['z'] = self.weight1 * x1 + self.weight2 * x2 + self.bias
        self.compute_grad['out'] = self.sigmoid(self.weight1 * x1 + self.weight2 * x2 + self.bias)
        return self.compute_grad['out']

    def clip(self, x, left, right):
        if x <= left:
            return left
        elif x >= right:
            return right
        else:
            return x

    def compute_loss(self, label, logit):
        logit = self.clip(logit, 1e-5, 1-1e-5)
        losses = -(label * math.log(logit) + (1-label) * math.log(1-logit))

        
        print('loss', losses)
        return logit - label, losses

    def backward(self, loss):
        self.grad_weight1 = loss * self.compute_grad['x1']
        self.grad_weight2 = loss  * self.compute_grad['x2']
        self.grad_bias = loss
        print(self.grad_weight1, self.grad_weight2, self.grad_bias)
        return
    
    def update(self):
        self.weight1 -= self.alpha * self.grad_weight1
        self.weight2 -= self.alpha * self.grad_weight2
        self.bias -= self.alpha * self.grad_bias

        print('update:', self.weight1, self.weight2, self.bias)
        return 
    
    def uniform(self, x1, x2, y):
        for x11, x12, yy in zip(x1, x2, y):
            loss, losses = self.compute_loss(yy, self.forward(x11, x12))
            self.backward(loss)
            self.update()
    
        return [self.weight1, self.weight2, self.bias]

lr = LogisticRegression()
x1 = [random.random() for _ in range(10000)]
x2 = [random.random() for _ in range(10000)]
y = [1 if 1.3*xx + 1.8*xx2 + 1.39 > 0.5 else 0 for xx, xx2 in zip(x1, x2)]

weight = lr.uniform(x1, x2, y)
print(weight)