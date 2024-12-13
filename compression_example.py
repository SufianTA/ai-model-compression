# ai-model-compression/compression_example.py

import torch
import torch.nn as nn

# Simple model compression (pruning example)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# Pruning the model
model = SimpleModel()
pruned_model = torch.nn.utils.prune.l1_unstructured(model.fc, name="weight", amount=0.2)

# Example forward pass
input_tensor = torch.randn(1, 10)
output = pruned_model(input_tensor)
print(output)
