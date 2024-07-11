# AixrTensor

 -Aixr Tensor is a powerful and flexible deep learning framework designed to optimize neural network training and inference. By leveraging dynamic device management, advanced optimization techniques, and custom functions, Aixr Tensor aims to provide an efficient and user-friendly environment for deep learning practitioners.

# Usage

 - Here is an example of how to use AixrTensor to create a simple neural network:

```python
import numpy as np
from tensor import Tensor
from optim import Aixr
from model import NeuralNetwork
from custom_functions import example_custom_func, example_important_func

# Define model layers (example)
layers = [
    # Add layers here, e.g., Linear, ReLU
]

# Create NeuralNetwork instance
model = NeuralNetwork(layers)

# Define optimizer with custom functions
optimizer = Aixr(model.parameters(), lr=0.001, custom_func=example_custom_func, important_func=example_important_func)

# Training loop
for epoch in range(epochs):
    # Forward pass
    output = model.forward(input_data)
    
    # Calculate loss
    loss = loss_function(output, target_data)
    
    # Backward pass
    model.backward(loss.gradient())
    
    # Optimization step
    optimizer.step()
    
    # Zero gradients
    optimizer.zero_grad()

    # Process important data
    model.process_important_data(example_important_func)


