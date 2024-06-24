import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, dtype=np.float32):
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad
        if self._grad_fn:
            self._grad_fn.backward(grad)

    def __add__(self, other):
        return Add.apply(self, other)

    def __matmul__(self, other):
        return MatMul.apply(self, other)
