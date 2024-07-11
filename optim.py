# optim.py
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
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, power=2, threshold=0.8, custom_func=None, important_func=None):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.power = power
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0
        self.threshold = threshold
        self.custom_func = custom_func
        self.important_func = important_func

    def step(self):
        self.t += 1
        lr_t = self.lr * (1 - self.t / 10000)  # Dynamic learning rate adjustment

        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                if self.important_func and self.important_func(param.data):
                    param.to('gpu')
                else:
                    param.mixed_device_operation(self.threshold, self.custom_func)
                    
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
                param.data -= lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

class AdamW:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, custom_func=None, important_func=None):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0
        self.custom_func = custom_func
        self.important_func = important_func

    def step(self):
        self.t += 1
        lr_t = self.lr * (1 - self.t / 10000)  # Dynamic learning rate adjustment

        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                if self.important_func and self.important_func(param.data):
                    param.to('gpu')
                else:
                    param.mixed_device_operation(self.threshold, self.custom_func)
                
                param.data -= self.weight_decay * param.data
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (param.grad ** 2)
                m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
                v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
                param.data -= lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

class LAMB:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, custom_func=None, important_func=None):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0
        self.custom_func = custom_func
        self.important_func = important_func

    def step(self):
        self.t += 1
        lr_t = self.lr * (1 - self.t / 10000)  # Dynamic learning rate adjustment

        for i, param in enumerate(self.parameters):
            if param.requires_grad and param.grad is not None:
                if self.important_func and self.important_func(param.data):
                    param.to('gpu')
                else:
                    param.mixed_device_operation(self.threshold, self.custom_func)
                
                param.data -= self.weight_decay * param.data
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (param.grad ** 2)
                m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
                v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
                r1 = np.linalg.norm(param.data)
                r2 = np.linalg.norm(m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * param.data)
                trust_ratio = r1 / r2 if r1 != 0 and r2 != 0 else 1.0
                param.data -= lr_t * trust_ratio * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * param.data)

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None
