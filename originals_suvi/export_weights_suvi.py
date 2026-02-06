"""Export PyTorch weights as raw float32 binary files for Fortran consumption.

This script bridges the Python/PyTorch world and the Fortran inference
implementation.  It performs two jobs:

1. **Weight export** -- Each linear layer's weight matrix and bias vector are
   written to separate ``.bin`` files in the ``weights/`` directory.  Weight
   matrices are transposed from PyTorch's ``[out_features, in_features]``
   layout to ``[in_features, out_features]`` and serialised in **Fortran
   column-major** order so that a simple ``read(u) array`` in Fortran fills
   the array correctly.

2. **Test-vector generation** -- Several input vectors (random with fixed
   seeds + edge cases) are fed through the PyTorch model.  Both the inputs
   and the resulting outputs are saved as raw float32 ``.bin`` files.  The
   Fortran test program reads these to validate its own forward pass against
   the Python reference.

Output files
------------
All files are written to the ``weights/`` directory:

Weights (raw float32):
    ``weight_0.bin``  -- Layer 0 weights, shape (50, 512)  in Fortran order
    ``bias_0.bin``    -- Layer 0 biases,  shape (512,)
    ``weight_1.bin``  -- Layer 1 weights, shape (512, 512) in Fortran order
    ``bias_1.bin``    -- Layer 1 biases,  shape (512,)
    ``weight_2.bin``  -- Layer 2 weights, shape (512, 3)   in Fortran order
    ``bias_2.bin``    -- Layer 2 biases,  shape (3,)

Test vectors (raw float32):
    ``test_input_N.bin``   -- Input vector  of shape (50,) for test case N
    ``test_output_N.bin``  -- Reference output of shape (3,) for test case N

Metadata:
    ``num_tests.txt``  -- Single integer: total number of test cases

Usage::

    conda activate pytorch
    python export_weights.py

Requirements:
    ``network.pth`` must exist in the working directory.
    ``nn_inference_minimal.py`` is imported for the ``Net`` class and
    ``remap_state_dict`` helper.
"""

import os
import torch
import numpy as np

from nn_inference_minimal import Net, remap_state_dict

OUT_DIR = "weights"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = Net()
sd = torch.load("./network.pth", map_location="cpu")
model.load_state_dict(remap_state_dict(sd))
model.eval()

# ---------------------------------------------------------------------------
# Export weights
# ---------------------------------------------------------------------------
# Each tuple: (state_dict key prefix, weight filename, bias filename)
layer_keys = [
    ("layers.linear_0", "weight_0", "bias_0"),
    ("layers.linear_1", "weight_1", "bias_1"),
    ("layers.linear_output", "weight_2", "bias_2"),
]

for orig_prefix, wfile, bfile in layer_keys:
    w = sd[f"{orig_prefix}.weight"].numpy()  # shape [out, in]
    b = sd[f"{orig_prefix}.bias"].numpy()

    # PyTorch stores weights as [out_features, in_features] in row-major (C)
    # order.  Fortran reads into an array declared as w(in, out) using
    # column-major order.  We transpose to (in, out) and write in F-order so
    # that the Fortran stream read fills the array correctly.
    w_f = np.asfortranarray(w.T, dtype=np.float32)  # shape (in, out), F-order
    with open(os.path.join(OUT_DIR, f"{wfile}.bin"), "wb") as f:
        f.write(w_f.tobytes(order="F"))
    b.astype(np.float32).tofile(os.path.join(OUT_DIR, f"{bfile}.bin"))
    print(f"  {wfile}: {w.shape} -> Fortran (in,out)={w_f.shape}")
    print(f"  {bfile}: {b.shape}")

# ---------------------------------------------------------------------------
# Generate test cases
# ---------------------------------------------------------------------------
test_cases = []

# Random vectors with fixed seeds for reproducibility
for seed in [42, 0, 123, 999, 7]:
    np.random.seed(seed)
    test_cases.append((f"random_seed{seed}", np.random.rand(50).astype(np.float32)))

# Edge cases
test_cases.append(("zeros", np.zeros(50, dtype=np.float32)))
test_cases.append(("ones", np.ones(50, dtype=np.float32)))
test_cases.append(("negative", -np.ones(50, dtype=np.float32)))

num_tests = len(test_cases)

with open(os.path.join(OUT_DIR, "num_tests.txt"), "w") as f:
    f.write(f"{num_tests}\n")

print(f"\nGenerating {num_tests} test cases:")
for i, (name, inp) in enumerate(test_cases):
    inp_tensor = torch.tensor(inp)
    with torch.no_grad():
        out = model(inp_tensor).numpy()

    inp.tofile(os.path.join(OUT_DIR, f"test_input_{i}.bin"))
    out.tofile(os.path.join(OUT_DIR, f"test_output_{i}.bin"))
    print(f"  [{i}] {name:20s}  output: {out}")

print(f"\nAll files written to {OUT_DIR}/")
