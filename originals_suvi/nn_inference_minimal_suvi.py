"""Minimal PyTorch inference for a 3-layer MLP (50 -> 512 -> 512 -> 3).

Loads a trained state_dict from ``network.pth`` and runs a single forward
pass on a random input vector.  The network uses ReLU activations between
hidden layers and no activation on the output layer.

Usage::

    conda activate pytorch
    python nn_inference_minimal.py

The state_dict stored in ``network.pth`` uses named layer keys
(``layers.linear_0``, ``layers.linear_1``, ``layers.linear_output``),
which differ from the ``nn.Sequential`` index-based keys used by the
``Net`` class defined here.  The ``remap_state_dict`` function translates
between the two naming conventions.
"""

import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    """Three-layer MLP matching the architecture in ``network.pth``.

    Architecture::

        Input (50) -> Linear(50, 512) -> ReLU
                   -> Linear(512, 512) -> ReLU
                   -> Linear(512, 3)   -> Output

    The ``nn.Sequential`` wrapper assigns integer indices to each sub-layer:
    0 = Linear, 1 = ReLU, 2 = Linear, 3 = ReLU, 4 = Linear.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(50, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

    def forward(self, x):
        """Run a forward pass through all layers.

        Args:
            x: Input tensor of shape ``(50,)`` or ``(batch, 50)``.

        Returns:
            Output tensor of shape ``(3,)`` or ``(batch, 3)``.
        """
        return self.layers(x)


def remap_state_dict(sd):
    """Translate saved state_dict keys to ``nn.Sequential`` index keys.

    The saved ``network.pth`` uses descriptive names::

        layers.linear_0.weight      -> layers.0.weight
        layers.linear_0.bias        -> layers.0.bias
        layers.linear_1.weight      -> layers.2.weight   (index 2, after ReLU at 1)
        layers.linear_1.bias        -> layers.2.bias
        layers.linear_output.weight -> layers.4.weight   (index 4, after ReLU at 3)
        layers.linear_output.bias   -> layers.4.bias

    Args:
        sd: ``OrderedDict`` loaded from ``network.pth``.

    Returns:
        New ``dict`` with keys matching the ``Net`` model definition.
    """
    mapping = {
        "layers.linear_0.weight": "layers.0.weight",
        "layers.linear_0.bias": "layers.0.bias",
        "layers.linear_1.weight": "layers.2.weight",
        "layers.linear_1.bias": "layers.2.bias",
        "layers.linear_output.weight": "layers.4.weight",
        "layers.linear_output.bias": "layers.4.bias",
    }
    return {mapping[k]: v for k, v in sd.items()}


if __name__ == "__main__":
    model = Net()
    sd = torch.load("./network.pth", map_location="cpu")
    model.load_state_dict(remap_state_dict(sd))
    model.eval()

    np.random.seed(42)
    test_input = torch.tensor(np.random.rand(50).astype(np.float32))

    with torch.no_grad():
        output = model(test_input)

    print("Input shape:", test_input.shape)
    print("Output:", output.numpy())
