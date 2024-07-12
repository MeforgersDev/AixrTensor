import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, dtype=np.float32, device='cpu'):
        self.device = device
        if self.device == 'tpu':
            import jax.numpy as jnp
            self.data = jnp.array(data, dtype=dtype)
        elif self.device == 'gpu':
            import cupy as cp
            self.data = cp.array(data, dtype=dtype)
        else:
            self.data = np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        self._original_data = self.data

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, device={self.device})"

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

    def to(self, device):
        self.device = device
        if device == 'tpu':
            import jax.numpy as jnp
            self.data = jnp.array(self._original_data)
        elif device == 'gpu':
            import cupy as cp
            self.data = cp.array(self._original_data)
        else:
            self.data = np.array(self._original_data)
        return self

    def save_to_ram(self):
        self._original_data = np.array(self.data)

    def load_from_ram(self):
        if self.device == 'tpu':
            import jax.numpy as jnp
            self.data = jnp.array(self._original_data)
        elif self.device == 'gpu':
            import cupy as cp
            self.data = cp.array(self._original_data)
        else:
            self.data = np.array(self._original_data)

    def mixed_device_operation(self, threshold=0.8, custom_func=None):
        import psutil
        ram_usage = psutil.virtual_memory().percent
        if custom_func:
            custom_func(self, ram_usage)
        elif ram_usage > threshold * 100:
            self.save_to_ram()
            self.to('gpu')
        else:
            self.load_from_ram()
            self.to('cpu')

    def process_important_data(self, is_important_func):
        if is_important_func(self.data):
            self.to('gpu')
        else:
            self.to('cpu')
