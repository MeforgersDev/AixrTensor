import numpy as np
from tensor import Tensor
from functions import Function

class Linear(Function):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = Tensor(np.random.randn(in_features, out_features) * 0.01, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, input):
        self.save_for_backward(input)
        return Tensor(np.dot(input.data, self.weights.data) + self.bias.data, requires_grad=True)

    def backward(self, grad):
        input, = self.saved_tensors
        if self.weights.requires_grad:
            self.weights.backward(np.dot(input.data.T, grad))
        if self.bias.requires_grad:
            self.bias.backward(np.sum(grad, axis=0, keepdims=True))
        if input.requires_grad:
            input.backward(np.dot(grad, self.weights.data.T))
