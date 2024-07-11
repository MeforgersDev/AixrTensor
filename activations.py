import numpy as np
from .tensor import Tensor
from .functions import Function

class ReLU(Function):
    @staticmethod
    def forward(x):
        result = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
        result._grad_fn = ReLU
        return result

    @staticmethod
    def backward(grad):
        x, = ReLU.saved_tensors
        grad_input = grad * (x.data > 0)
        if x.requires_grad:
            x.backward(grad_input)

class Sigmoid(Function):
    @staticmethod
    def forward(x):
        result = 1 / (1 + np.exp(-x.data))
        result = Tensor(result, requires_grad=x.requires_grad)
        result._grad_fn = Sigmoid
        return result

    @staticmethod
    def backward(grad):
        x, = Sigmoid.saved_tensors
        grad_input = grad * (1.0 - x.data) * x.data
        if x.requires_grad:
            x.backward(grad_input)

class Tanh(Function):
    @staticmethod
    def forward(x):
        result = np.tanh(x.data)
        result = Tensor(result, requires_grad=x.requires_grad)
        result._grad_fn = Tanh
        return result

    @staticmethod
    def backward(grad):
        x, = Tanh.saved_tensors
        grad_input = grad * (1 - np.square(x.data))
        if x.requires_grad:
            x.backward(grad_input)
