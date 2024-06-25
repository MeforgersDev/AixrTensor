# AixrTensor

- AixrTensor is a simple deep learning library built from scratch using NumPy. It includes basic tensor operations, automatic differentiation, and some neural network layers.

-Requirements

 - Numpy

-Usage

 - Here is an example of how to use AixrTensor to create a simple neural network:

```python
from AixrTensor import Tensor, Linear, ReLU, SGD

# Training data
x = Tensor([[1, 2], [2, 3], [3, 4], [4, 5]], requires_grad=True)
y = Tensor([[3], [5], [7], [9]], requires_grad=False)

# Model and optimizer
linear_layer = Linear(2, 1)
relu = ReLU()
optimizer = SGD([linear_layer.weights, linear_layer.bias], lr=0.01)

# Training loop
for epoch in range(1000):
    y_pred = linear_layer(x)
    y_pred = relu(y_pred)
    loss = Tensor(np.mean((y_pred.data - y.data) ** 2))

    # Backward pass
    loss.backward()

    # Optimization step
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.data}')

# Prediction
y_pred = linear_layer(x)
print(f'Predicted: {y_pred.data}')
