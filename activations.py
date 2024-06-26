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
