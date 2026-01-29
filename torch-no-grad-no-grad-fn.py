"""
Understanding the error:

RuntimeError: element 0 of variables does not require grad and does not have a grad_fn
"""

import torch

# 1. A simple model
model = torch.nn.Linear(5, 1)
input_data = torch.randn(1, 5)

# 2. Forward pass
output = model(input_data)

# 3. BREAK THE GRAPH (The "Direct" way)
# By calling .detach(), we create a view that doesn't track history.
# Or, by performing an operation on the underlying data:
loss = output.detach().mean()

print(f"Is grad tracked? {loss.requires_grad}") # False
print(f"Is there a grad_fn? {loss.grad_fn}")   # None

# 4. This will RAISE the exception:
# "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
loss.backward()
