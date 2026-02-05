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
import torch.nn as nn

from nn_inference_minimal import Net

store_in = Path("./weights")
store_in.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
sd = torch.load('./network.pth', map_location='cpu')

inp_size = sd['layers.linear_0.weight'].shape[1]
# We have two entries (weight and bias) per layer, plus one output layer
nlayers_hidden = len(sd.keys())//2 - 1
# Infer number of neurons per layer from weight shapes
nneurons = [sd[f'layers.linear_{i}.weight'].shape[0] for i in range(nlayers_hidden)]
outp_size = sd['layers.linear_output.weight'].shape[0]

nneurons_full = [inp_size] + nneurons + [outp_size]

# Export nneurons_full array to binary file for Fortran code to read
nneurons_array = np.array(nneurons_full, dtype=np.int32)
with open(store_in / 'nneurons.bin', 'wb') as f:
    f.write(nneurons_array.tobytes())

# Save number of layers to binary file for Fortran code to read
with open(store_in / 'nlayers.bin', 'wb') as f:
    f.write(np.array([nlayers_hidden+1], dtype=np.int32).tobytes())

network = Net(inp_size=inp_size, nlayers=nlayers_hidden, nneurons=nneurons, outp_size=outp_size)

network.load_state_dict(sd)
network.eval()

n = m = 0 # counts number of layers which are not activation layers and which are activation layers, respectively
for i, l in enumerate(network.layers):
    # Check which activations are used in the network
    if isinstance(l, nn.Linear):
        print(f"Linear layer found in the network at {n}.")
        w_key = f'layers.linear_{n}.weight' if n < nlayers_hidden else 'layers.linear_output.weight'
        b_key = f'layers.linear_{n}.bias' if n < nlayers_hidden else 'layers.linear_output.bias'

        w_pt = sd[w_key].numpy().astype(np.float32)
        b_pt = sd[b_key].numpy().astype(np.float32)

        print(f'Size of weights in PyTorch (out, in) at network layer {n}:', w_pt.shape)

        # We need to transform w and b in two ways so Fortran can read them correctly:
        # 1. Reverse the (output, input) shape to (input, output) by transposing
        # 2. Save in column-major order (Fortran order)

        w_f = np.asfortranarray(w_pt.T, dtype=np.float32)
        b_f = np.asfortranarray(b_pt, dtype=np.float32)

        wname = f'layer_{n}_weights'
        bname = f'layer_{n}_biases'

        with open(f'{store_in}/{wname}.bin', 'wb') as f:
            f.write(w_f.tobytes(order='F'))
        with open(f'{store_in}/{bname}.bin', 'wb') as f:
            f.write(b_f.tobytes(order='F'))
        n += 1
        # Do nothing else
    elif isinstance(l, nn.ReLU):
        if m == n-1:
            print(f"ReLU activation found in the network at {m}.")
            with open(store_in / f'layer_{m}_act.txt', 'w') as f:
                f.write('relu')
            m += 1
    else:
        raise ValueError(f"Unknown layer type {type(l)} found in the network at {i}.")

if m == n-1:
    print('Last layer has no activation function, as expected.')
    with open(store_in / f'layer_{m}_act.txt', 'w') as f:
        f.write('id')