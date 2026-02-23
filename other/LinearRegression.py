# 手写一个线性回归
import math
import random

class LinearRegression:
    def __init__(self):
        self.weight1 = 0.1
        self.weight2 = 0.1
        self.bias = 0.0

        self.grad_weight1 = 0
        self.grad_weight2 = 0
        self.grad_bias = 0


        self.alpha = 0.01

        self.compute_grad = {'x1':0, 'x2':0}
    
    def forward(self, x1, x2):
        self.compute_grad['x1'] = x1
        self.compute_grad['x2'] = x2
        return self.weight1 * x1 + self.weight2 * x2 + self.bias

    def compute_loss(self, label, logit):
        losses = (logit - label) ** 2
        print('loss', losses)
        return (logit - label), losses

    def backward(self, loss):
        self.grad_weight1 = 2 * loss * self.compute_grad['x1']
        self.grad_weight2 = 2 * loss  * self.compute_grad['x2']
        self.grad_bias = 2 * loss
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
            print('loss', losses)
            self.backward(loss)
            self.update()
    
        return [self.weight1, self.weight2, self.bias]

lr = LinearRegression()
x1 = [random.random() for _ in range(10000)]
x2 = [random.random() for _ in range(10000)]
y = [1.3*xx + 1.8*xx2 + 1.39 for xx, xx2 in zip(x1, x2)]

weight = lr.uniform(x1, x2, y)
print(weight)