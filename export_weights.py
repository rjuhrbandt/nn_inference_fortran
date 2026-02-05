'''
PyTorch inference for MLPs and export of weights and test cases.

Steps:
1. Load state_dict from a trained model file "network.pth" and perform forward pass.
2. Export weights and biases as raw float32 binary files in Fortran order (PyTorch uses row-major order by default
   while Fortran uses column-major order, so we need to transpose them).
3. Generate test cases including edge cases and actual evaluation inputs from 
   the testing dataset used during offline evaluation, saving them as raw float32 binary files.

Assumptions:
- The network is linear and uses fully connected layers only.
- All hidden layers (except output layer) use ReLU activation. Output layer has no activation.

'''

import os
from pathlib import Path
import torch
import numpy as np

from nn_inference_minimal import Net

store_in = Path("./weights")
store_in.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Export weights from state dict
# ---------------------------------------------------------------------------

# Iterate through number of layers to export weights and biases
for i in range(nlayers + 1):
    w_key = f'layers.linear_{i}.weight' if i < nlayers else 'layers.linear_output.weight'
    b_key = f'layers.linear_{i}.bias' if i < nlayers else 'layers.linear_output.bias'

    w_pt = sd[w_key].numpy().astype(np.float32)
    b_pt = sd[b_key].numpy().astype(np.float32)

    # We need to transform w and b in two ways so Fortran can read them correctly:
    # 1. Reverse the (output, input) shape to (input, output) by transposing
    # 2. Save in column-major order (Fortran order)

    w_f = np.asfortranarray(w_pt.T, dtype=np.float32)
    b_f = np.asfortranarray(b_pt, dtype=np.float32)

    wname = f'layer_{i}_weights'
    bname = f'layer_{i}_biases'

    with open(f'{store_in}/{wname}.bin', 'wb') as f:
        f.write(w_f.tobytes(order='F'))
    b_f.tofile(f'{store_in}/{bname}.bin')