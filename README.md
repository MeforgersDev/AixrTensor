# AixrTensor

 - This framework is a powerful tool for understanding, developing, and optimizing deep learning models. Its flexible structure makes it suitable for various research and development projects. It is also an excellent resource for those who want to grasp the fundamentals of deep learning algorithms. With GitHub integration, you can easily share and collaborate on your project.

# Usage

 - Here is an example of how to use AixrTensor to create a simple neural network:

```python
from AixrTensor import Tensor, Linear, ReLU, Aixr, NeuralNetwork

# Create a simple model
model = NeuralNetwork([
    Linear(2, 4),
    ReLU(),
    Linear(4, 1)
])

# Define a loss function and optimizer
criterion = ...  # Define your loss function here
optimizer = Aixr(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, power=2)

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

# Example data_loader and criterion definitions
# data_loader = ...
# criterion = ...

# Training process
train(model, optimizer, data_loader, epochs=10)
