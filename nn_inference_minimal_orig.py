import torch
import numpy as np

# First load the trained network: Assuming it is in the same folder as this script
network = torch.load('./network.pth', map_location='cpu')

# Replace with actual data
test_ds = torch.Tensor(np.random.rand(50))

with torch.no_grad():
    network.eval()
    predictions = network(test_ds)
