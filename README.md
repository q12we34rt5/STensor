# STensor

STensor (Tensor from Scratch) is a lightweight deep learning framework based on NumPy, with a similar interface to PyTorch. Its goal is to implement automatic differentiation through concise code and serve as a teaching tool.

## Example

This example demonstrates how to use this framework to solve the XOR problem.

### Import Packages

```python
import stensor
import numpy as np
import time
```

### XOR Dataset

```python
def generate_XOR():
    x, y = np.mgrid[-1:1.1:0.1, -1:1.1:0.1].reshape((2, -1))

    quadrant_I   = (x > 0) & (y > 0)
    quadrant_III = (x < 0) & (y < 0)
    labels = quadrant_I | quadrant_III | 0

    return np.stack([x, y], axis=1), labels.reshape((-1, 1))
```

### Calculate Accuracy

```python
def accuracy(y_true, y_pred):
    return ((y_pred > 0.5) == y_true).sum() / y_true.shape[0]
```

### Network

```python
class Network(stensor.Module):

    def __init__(self, in_features, out_features, hidden_features):
        super(Network, self).__init__()
        self.fc1 = stensor.Linear(in_features, hidden_features)
        self.act1 = stensor.Sigmoid()
        self.fc2 = stensor.Linear(hidden_features, out_features)
        self.act2 = stensor.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x
```

### Training

```python
# Prepare dataset
x, y = map(stensor.create, generate_XOR())

# Model
net = Network(2, 1, 10)
print(net)
print('================================================')

# Training
optimizer = stensor.Adam(net.parameters(), lr=0.003)
lossfn = stensor.MSELoss()
epochs = 10000

total_time = 0
for epoch in range(epochs):
    start_time = time.time()

    # Forward and then backpropagation
    y_pred = net.forward(x)
    loss = lossfn.forward(y_pred, y)
    loss.backward()

    # Update model coefficients
    optimizer.step()

    # Reset gradients
    optimizer.zero_grad()

    end_time = time.time()
    total_time += end_time - start_time

    if (epoch + 1) % 1000 == 0:
        print(f'epoch: {epoch + 1}, loss: {loss.data:.4f}, accuracy: {accuracy(y.data, y_pred.data):.4f}')

print('================================================')
print(f'Time cost: {total_time}s')
```

### Output

```
Network(
  (fc1): Linear(in_features=2, out_features=10)
  (act1): Sigmoid()
  (fc2): Linear(in_features=10, out_features=1)
  (act2): Sigmoid()
)
================================================
epoch: 1000, loss: 32.0482, accuracy: 0.9252
epoch: 2000, loss: 16.9099, accuracy: 0.9773
epoch: 3000, loss: 11.6299, accuracy: 0.9887
epoch: 4000, loss: 8.3179, accuracy: 0.9955
epoch: 5000, loss: 5.5203, accuracy: 0.9955
epoch: 6000, loss: 3.5380, accuracy: 0.9955
epoch: 7000, loss: 2.1868, accuracy: 0.9955
epoch: 8000, loss: 1.2476, accuracy: 1.0000
epoch: 9000, loss: 0.6771, accuracy: 1.0000
epoch: 10000, loss: 0.3690, accuracy: 1.0000
================================================
Time cost: 2.398193120956421s
```
