# PyTorch-to-Fortran Inference Pipeline

## Overview

This pipeline ports a trained PyTorch MLP to native Fortran for
neural-network inference. It is a two-stage process:

1. **Python** exports the raw binary weights from a trained model.
2. **Fortran** loads those weights and runs inference using `matmul` plus
   element-wise activation — no Python, no external ML libraries at
   runtime.

The network is a 3-layer MLP:

```
input(50) → Linear+ReLU(512) → Linear+ReLU(512) → Linear(3) → output(3)
```

---

## Why not load the model directly in Fortran?

PyTorch offers several serialisation formats, but none map cleanly to
Fortran:

| Format | What it saves | Why it doesn't help |
|---|---|---|
| `state_dict` (`.pth`) | Learnable parameters only (weights, biases) | No architecture, no activation functions |
| ONNX | Full computation graph | No native Fortran reader |
| TorchScript | Full model in PyTorch's own IR | Format is PyTorch-specific |

Libraries such as **FTorch** and **neural-fortran** can bridge the gap,
but they introduce external dependencies. The raw-binary approach used
here is dependency-free and proven correct by cross-validation.

---

## Step 1 — Python exports weights (`export_weights.py`)

The script loads the checkpoint `network.pth`, iterates over the three
linear layers, and writes each weight matrix and bias vector as raw
IEEE 754 float32 bytes.

### Weight transposition

PyTorch stores weight matrices in **row-major** layout with shape
`[out_features, in_features]`. Fortran arrays are **column-major**. To
make the Fortran `read` fill the array correctly, the export transposes
every weight matrix before writing:

```python
# PyTorch shape: (out, in) row-major
# After .T and .astype(np.float32).tobytes(): (in, out) column-major
weight.T.astype(np.float32).tofile(path)
```

Bias vectors are 1-D, so they are written directly — no ordering
ambiguity.

### Output files

All files are written to the `weights/` directory:

| File | Shape (Fortran) | Description |
|---|---|---|
| `weight_0.bin` | (50, 512) | Layer 0 weights |
| `bias_0.bin` | (512) | Layer 0 biases |
| `weight_1.bin` | (512, 512) | Layer 1 weights |
| `bias_1.bin` | (512) | Layer 1 biases |
| `weight_2.bin` | (512, 3) | Layer 2 weights |
| `bias_2.bin` | (3) | Layer 2 biases |
| `test_input_N.bin` | (50) | Test input vector N |
| `test_output_N.bin` | (3) | Reference output vector N |
| `num_tests.txt` | — | Number of test cases (plain text) |

Eight test cases are generated: five from fixed random seeds
(42, 0, 123, 999, 7) and three edge cases (zeros, ones, negative ones).

---

## Step 2 — Fortran loads raw bytes (`nn_module.f90`)

The module declares arrays matching the exported shapes:

```fortran
real(4) :: w0(INPUT_DIM, HIDDEN_DIM)   ! (50, 512)
real(4) :: b0(HIDDEN_DIM)              ! (512)
real(4) :: w1(HIDDEN_DIM, HIDDEN_DIM)  ! (512, 512)
real(4) :: b1(HIDDEN_DIM)              ! (512)
real(4) :: w2(HIDDEN_DIM, OUTPUT_DIM)  ! (512, 3)
real(4) :: b2(OUTPUT_DIM)              ! (3)
```

Loading uses Fortran stream I/O:

```fortran
open(newunit=u, file=fname, access='stream', form='unformatted', ...)
read(u) array
```

`access='stream'` means no Fortran record headers or markers — the file
is pure bytes, exactly as Python wrote them. A single `read(u) w0` fills
the entire `(50, 512)` array in column-major order, which is correct
because the export already transposed the matrix into that layout.

---

## Step 3 — Fortran forward pass

The `forward` subroutine performs inference in three lines of arithmetic:

```fortran
h1 = matmul(input, w0) + b0;  h1 = max(h1, 0.0)   ! ReLU
h2 = matmul(h1, w1) + b1;     h2 = max(h2, 0.0)    ! ReLU
output = matmul(h2, w2) + b2                         ! no activation
```

### Why `matmul(input, W)` works

In PyTorch the linear layer computes `output = input @ W^T + b` where
`W` has shape `[out, in]`. The export step transposes `W` to
`[in, out]`, so the stored matrix is already `W^T`. Therefore the
Fortran expression `matmul(input, w)` is equivalent to
`input @ W_original^T` — identical to the PyTorch computation.

### ReLU activation

ReLU is implemented with Fortran's intrinsic `max`:

```fortran
h1 = max(h1, 0.0)
```

This is applied element-wise because `h1` is an array and `0.0` is a
scalar.

---

## Cross-validation (`main.f90`)

The test program validates that the Fortran forward pass reproduces
PyTorch's output:

1. Load weights from `weights/`.
2. Read the test count from `weights/num_tests.txt`.
3. For each test case:
   - Read the input vector from `weights/test_input_N.bin`.
   - Read the Python reference output from `weights/test_output_N.bin`.
   - Run the Fortran forward pass.
   - Compute `max_err = maxval(abs(output - ref_output))`.
   - Report **PASS** if `max_err < 1.0e-5`, otherwise **FAIL**.
4. Exit with code 1 if any test failed.

In practice, observed errors are on the order of 1e-8, well below the
1e-5 threshold. This confirms that the weight export and Fortran
inference are numerically consistent with PyTorch.

---

## Caveats

- **Activation function is assumed.** The `state_dict` stores only
  weights and biases — it does not record which activation function was
  used between layers. The Fortran code hard-codes ReLU (`max(x, 0.0)`).
  If training used a different activation (e.g. GELU, Swish), both the
  Python reference forward pass and the Fortran code must be updated to
  match.

- **Cross-validation proves porting correctness, not activation
  correctness.** Both sides use the same activation, so agreement only
  shows that the port is faithful. It does not verify that the activation
  matches the one used during training — that must be confirmed by
  inspecting the model definition.
