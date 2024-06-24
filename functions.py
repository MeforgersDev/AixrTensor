import numpy as np
from .tensor import Tensor

class Function:
    def __init__(self):
        self.saved_tensors = []

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    @classmethod
    def apply(cls, *args):
        obj = cls()
        result = obj.forward(*args)
        result._grad_fn = obj
        return result

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

class Add(Function):
    @staticmethod
    def forward(a, b):
        result = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)
        result._grad_fn = Add
        return result

    @staticmethod
    def backward(grad):
        a, b = Add.saved_tensors
        if a.requires_grad:
            a.backward(grad)
        if b.requires_grad:
            b.backward(grad)

class MatMul(Function):
    @staticmethod
    def forward(a, b):
        result = Tensor(np.matmul(a.data, b.data), requires_grad=a.requires_grad or b.requires_grad)
        result._grad_fn = MatMul
        return result

    @staticmethod
    def backward(grad):
        a, b = MatMul.saved_tensors
        if a.requires_grad:
            a.backward(np.matmul(grad, b.data.T))
        if b.requires_grad:
            b.backward(np.matmul(a.data.T, grad))
