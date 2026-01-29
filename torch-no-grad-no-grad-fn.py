"""
Understanding the error:

RuntimeError: element 0 of variables does not require grad and does not have a grad_fn
"""

import torch
import numpy as np

# 1. Setup a simple linear model
model = torch.nn.Linear(10, 1)
input_data = torch.randn(1, 10)
target = torch.tensor([[1.0]])

# 2. Forward pass
output = model(input_data)

# 3. BREAK THE GRAPH
# Converting to numpy and back to a tensor creates a NEW tensor
# that has no history of the 'model' operations.
output_numpy = output.detach().numpy()
loss = torch.tensor((output_numpy - target.numpy())**2, requires_grad=True)

# 4. Attempt backward pass
print(f"Loss grad_fn: {loss.grad_fn}") # This will be None
loss.backward() # This will trigger your exception
