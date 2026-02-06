'''
Minimal version of PyTorch inference for MLPs.

Steps:
1. Load state_dict from a trained model file "network.pth".
2. Define the network architecture matching the saved model. 
3. Perform the forward pass.

Assumptions:
- The network is linear and uses fully connected layers only.
- All hidden layers (except output layer) use ReLU activation. Output layer has no activation.

'''

import torch
import torch.nn as nn
import numpy as np
import sys

class Net(nn.Module):
    """ 
    Simple MLP with dynamic layer creation.

    Assumptions:
        - All hidden layers (except the output layer) use ReLU activation. Output layer uses no activation.

    Args:
        inp_size (int): Size of the input layer.
        nlayers (int): Number of hidden layers.
        nneurons (int or list of int): Number of neurons in each hidden layer. If an int is provided, all hidden layers will have the same number of neurons.
        outp_size (int): Size of the output layer.
    
    """
    def __init__(self, inp_size, nlayers, nneurons, outp_size):
        super(Net, self).__init__()
        self.layers = nn.Sequential()

        # Dynamically create layers
        self.layers.add_module(f'linear_0', nn.Linear(inp_size, nneurons[0]))
        self.layers.add_module(f'relu_0', nn.ReLU())

        if nlayers > 1:
            for i in range(1, nlayers): # Start from second layer, index 1
                self.layers.add_module(f"linear_{i}", nn.Linear(nneurons[i-1], nneurons[i]))
                self.layers.add_module(f"relu_{i}", nn.ReLU())
        self.layers.add_module(f"linear_output", nn.Linear(nneurons[-1], outp_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network ('prediction').
        """
        y = self.layers(x)

        return y

if __name__ == "__main__":
    sd = torch.load('./network.pth', map_location='cpu')

    inp_size = sd['layers.linear_0.weight'].shape[1]
    # We have two entries (weight and bias) per layer, plus one output layer
    nlayers = len(sd.keys())//2 - 1
    # Infer number of neurons per layer from weight shapes
    nneurons = [sd[f'layers.linear_{i}.weight'].shape[0] for i in range(nlayers)]
    outp_size = sd['layers.linear_output.weight'].shape[0]

    network = Net(inp_size=inp_size, nlayers=nlayers, nneurons=nneurons, outp_size=outp_size)
    network.load_state_dict(sd)
    network.eval()