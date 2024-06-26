# AixrTensor

 - AixrTensor is a simple deep learning library built from scratch using NumPy. It includes basic tensor operations, automatic differentiation, and some neural network layers.

# Requirements

 - Numpy

# Usage

 - Here is an example of how to use AixrTensor to create a simple neural network:

```python
from my_ai_framework import Tensor, Linear, ReLU, Aixr, NeuralNetwork

# Create a simple model
model = NeuralNetwork([
    Linear(2, 4),
    ReLU(),
    Linear(4, 1)
])

# Define a loss function and optimizer
criterion = ...  # Define your loss function here
optimizer = Aixr(model.parameters(), lr=0.01)

# Train the model
def train(model, optimizer, data_loader, epochs=10):
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            # Forward pass
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.data}')

# Example data loader and training loop
# data_loader = ...  # Define your data loader here
# train(model, optimizer, data_loader, epochs=10
