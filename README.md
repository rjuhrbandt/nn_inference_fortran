# Neural Network Inference: PyTorch to Fortran

Port of a trained PyTorch MLP to native Fortran for deployment without a
Python runtime.  Includes automated cross-validation between the two
implementations.

## Network Architecture

```
Input (50) ── Linear(50, 512) ── ReLU ── Linear(512, 512) ── ReLU ── Linear(512, 3) ── Output
```

- **3 fully-connected (linear) layers**, two hidden and one output
- **ReLU activation** (`max(x, 0)`) after each hidden layer
- **No activation** on the output layer
- All arithmetic is **single-precision float32**

The trained weights are stored in `network.pth` as a PyTorch `state_dict`
(an `OrderedDict` of tensors, not a serialised `nn.Module`).

## Setup

### Prerequisites

- conda or miniforge
- gfortran (tested with 15.2.0)
- GNU Make

### Create the conda environment

From the provided environment file:

```bash
conda env create -f environment.yml
conda activate pytorch
```

Or manually:

```bash
conda create -n pytorch python=3.11 -y
conda activate pytorch
pip install torch numpy
```

### Quick test — Python inference

```bash
python nn_inference_minimal.py
```

Expected output (values depend on the trained weights):

```
Input shape: torch.Size([50])
Output: [0.04143657 0.00477913 0.10192735]
```

## File Overview

| File                     | Purpose                                               |
|--------------------------|-------------------------------------------------------|
| `network.pth`            | Trained PyTorch state_dict                            |
| `environment.yml`        | Conda environment specification                       |
| `nn_inference_minimal.py`| Python model definition + single-sample inference     |
| `export_weights.py`      | Export weights to binary + generate test vectors      |
| `nn_module.f90`          | Fortran inference module (load weights, forward pass) |
| `main.f90`               | Fortran validation program                            |
| `Makefile`               | Build system for the Fortran code                     |

## How Weights Are Exported

### The problem: row-major vs column-major

PyTorch stores each linear layer's weight matrix in **row-major (C) order**
with shape `[out_features, in_features]`.  Fortran stores arrays in
**column-major order**.  A naive binary dump from Python would be
misinterpreted by a Fortran `read` statement because the two languages
traverse memory in opposite order.

### The solution

`export_weights.py` performs the following for each layer:

1. Read the PyTorch weight matrix `W` with shape `(out, in)`.
2. **Transpose** to `W^T` with shape `(in, out)`.
3. Convert to a **Fortran-order** (column-major) array via
   `np.asfortranarray`.
4. Write the raw bytes with `tobytes(order="F")`.

```python
w_f = np.asfortranarray(w.T, dtype=np.float32)   # shape (in, out), F-order
with open("weight_N.bin", "wb") as f:
    f.write(w_f.tobytes(order="F"))
```

On the Fortran side, the array is declared as `w(in_features, out_features)`
and read with stream I/O:

```fortran
real(4) :: w0(50, 512)
open(newunit=u, file='weight_0.bin', access='stream', form='unformatted')
read(u) w0
```

Because the file is written in column-major order and Fortran reads in
column-major order, the values land in the correct positions.

Bias vectors are 1-D so they have no ordering ambiguity — they are written
directly as contiguous float32 streams.

### Output files

All files are written to the `weights/` directory:

| File              | Contents                                  | Size (bytes)        |
|-------------------|-------------------------------------------|---------------------|
| `weight_0.bin`    | Layer 0 weights, shape (50, 512)          | 50 × 512 × 4       |
| `bias_0.bin`      | Layer 0 biases, shape (512,)              | 512 × 4             |
| `weight_1.bin`    | Layer 1 weights, shape (512, 512)         | 512 × 512 × 4      |
| `bias_1.bin`      | Layer 1 biases, shape (512,)              | 512 × 4             |
| `weight_2.bin`    | Layer 2 weights, shape (512, 3)           | 512 × 3 × 4        |
| `bias_2.bin`      | Layer 2 biases, shape (3,)                | 3 × 4               |
| `test_input_N.bin`| Test input vector, shape (50,)            | 50 × 4              |
| `test_output_N.bin`| Python reference output, shape (3,)      | 3 × 4               |
| `num_tests.txt`   | Number of test cases (plain text integer) | —                   |

### Assumptions

- The state_dict key prefix `layers.linear_0`, `layers.linear_1`,
  `layers.linear_output` maps to the first, second, and third linear layers
  respectively.
- Activation between hidden layers is **assumed to be ReLU**.  No activation
  is applied on the output layer.

  > **Caveat — activation function is not verifiable from the weights file.**
  > A PyTorch `state_dict` only stores learnable parameters (weights and
  > biases).  Parameterless activations such as ReLU, GELU, or Tanh leave
  > no trace in the `.pth` file, so the choice of ReLU cannot be confirmed
  > from `network.pth` alone.
  >
  > If the original training code used a different activation function, both
  > `nn_inference_minimal.py` and `nn_module.f90` would need to be updated
  > to match.
  >
  > The cross-validation (all 8 test cases passing at ~1e-8 error) proves
  > that the Python and Fortran implementations are **consistent with each
  > other**, but it does not prove that the assumed activation matches the
  > one used during training.
- All values are **IEEE 754 single-precision (float32)**.  No quantisation or
  mixed precision.
- Weights are exported for a **single-sample** forward pass (no batch
  dimension).

## How Test Cases Are Generated

`export_weights.py` creates **8 test cases** that exercise different regions
of the network's activation space:

| Index | Description           | How it is generated                              |
|-------|-----------------------|--------------------------------------------------|
| 0     | Random (seed 42)      | `np.random.seed(42); np.random.rand(50)`         |
| 1     | Random (seed 0)       | `np.random.seed(0); np.random.rand(50)`          |
| 2     | Random (seed 123)     | `np.random.seed(123); np.random.rand(50)`        |
| 3     | Random (seed 999)     | `np.random.seed(999); np.random.rand(50)`        |
| 4     | Random (seed 7)       | `np.random.seed(7); np.random.rand(50)`          |
| 5     | All zeros             | `np.zeros(50)`                                   |
| 6     | All ones              | `np.ones(50)`                                    |
| 7     | All negative ones     | `-np.ones(50)`                                   |

Each input is run through the PyTorch model in eval mode with
`torch.no_grad()`.  Both the input vector (50 floats) and the output vector
(3 floats) are written as raw float32 `.bin` files.  The total number of
test cases is written to `num_tests.txt` so the Fortran program doesn't need
to be recompiled when test cases change.

The edge cases are chosen deliberately:

- **All zeros**: tests the bias-only path (all pre-activations equal the
  biases, so the ReLU pattern depends entirely on the bias values).
- **All ones**: uniform positive input, stresses a different set of neurons.
- **All negative ones**: negative input values interact differently with the
  weight signs, likely producing a different ReLU activation pattern than the
  positive cases.

## How Fortran Loads Weights and Runs Inference

### Weight loading (`nn_module.f90`)

`load_weights('weights')` reads all 6 binary files into module-level arrays:

```fortran
real(4) :: w0(50,  512), b0(512)    ! Layer 0
real(4) :: w1(512, 512), b1(512)    ! Layer 1
real(4) :: w2(512, 3),   b2(3)      ! Layer 2
```

Each file is opened with `access='stream', form='unformatted'` (raw byte
stream, no Fortran record markers) and read in a single statement.

### Forward pass (`nn_module.f90`)

The `forward` subroutine implements:

```
h1     = ReLU( matmul(input, w0) + b0 )     -- 50  -> 512
h2     = ReLU( matmul(h1,    w1) + b1 )     -- 512 -> 512
output =       matmul(h2,    w2) + b2        -- 512 -> 3
```

`matmul(input, W)` with `input` as a rank-1 vector and `W` as a rank-2
array `(in, out)` produces a rank-1 result of size `out`.  This is
equivalent to PyTorch's `F.linear(x, W_original)` = `x @ W_original^T`,
since our stored `W` is already the transpose of the PyTorch weight.

ReLU is `max(h, 0.0)`, applied element-wise via Fortran's intrinsic `max`
on the whole array.

## Validation: Python vs Fortran

### Running the full pipeline

```bash
conda activate pytorch
python export_weights.py     # export weights + generate test vectors
make                         # compile Fortran
./nn_inference               # run validation
```

### What the Fortran program does

For each test case `i`:

1. Reads `weights/test_input_i.bin` (50 float32 values) — the same input
   that Python used.
2. Reads `weights/test_output_i.bin` (3 float32 values) — the Python
   reference output.
3. Runs the Fortran `forward()` subroutine on the input.
4. Computes `max_err = maxval(abs(fortran_output - python_output))` — the
   maximum absolute error across all 3 output elements.
5. Reports **PASS** if `max_err < 1e-5`, **FAIL** otherwise.

### Expected output

```
Running 8 test cases:
----------------------------------------------------------------------
Test case 0
  Python ref:     0.04143657    0.00477913    0.10192735
  Fortran:        0.04143658    0.00477913    0.10192735
  Max error:   7.45058E-09
  Result:     PASS
...
======================================================================
ALL TESTS PASSED
```

### Why the threshold is 1e-5

Both implementations use float32 arithmetic, but the order of
floating-point operations may differ slightly between PyTorch (which may use
fused multiply-add or different accumulation order) and Fortran's
`matmul` intrinsic.  In practice the observed errors are on the order of
**1e-8** (a few ULPs), well within the 1e-5 threshold.  The threshold is
deliberately conservative to avoid false failures from platform-specific
floating-point behaviour while still catching genuine logic errors.

### Exit code

The program exits with code **0** if all tests pass, **1** if any test fails.
This makes it usable in CI pipelines or automated scripts.
