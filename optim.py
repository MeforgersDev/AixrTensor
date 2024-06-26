import numpy as np

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

class Aixr:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, power=2):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.power = power
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                # Apply weight decay
                param.data -= self.weight_decay * param.data

                # Update biased first moment estimate
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
                
                # Update biased second raw moment estimate
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (param.grad ** self.power)

                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
                
                # Update parameters
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None
