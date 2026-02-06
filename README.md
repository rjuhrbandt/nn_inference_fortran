# Neural network inference - From Pytorch to Fortran

## Description

Combining Machine Learning (ML) with numerical ocean modelling is becoming more and more common nowadays. One significant challenge here is the porting across programming languages. ML infrastructure, such as neural networks, tend to be written in either Pytorch (Python) or Tensorflow/Keras. In numerical ocean models, however, Fortran remains a widely used language. 

This project provides a pipeline from the state dictionary of a neural network (multi-layer perceptron, MLP) in Pytorch (provided as a .pth file) to the inference using the state dictionary's weights and biases within a neural network module for Fortran. 

## File descriptions

| File                     | Purpose                                                                                                              |
|--------------------------|----------------------------------------------------------------------------------------------------------------------|
| `environment.yml`        | Environment file for running the Python files.                                                                       |
| `export_weights.py`      | Exporting the neural network infrastructure including weights and biases to binary files readable by Fortran.        |
| `main.f90`               | Fortran pipeline to read neural network parameters from binary files and and apply inference to several test cases.  |
| `network.pth`            | State dictionary (state_dict) of the trained Pytorch neural network.                                                 |
| `nn_inference_minimal.py`| Minimal version of PyTorch inference for MLP
| `nn_module.f90`          | Introduces a module containing the subroutines called in the main program.                                           |

## Prerequisites:
- conda or miniforge
- gfortran (tested with 15.2.0)
- GNU Make

1. Create the conda environment
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

2. Running Python inference with random numbers

```bash
python nn_inference_minimal.py
```

3. Running Fortran inference with test input and output
```bash
make clean
make
./nn_module
```

Output: PASS if abs(diff) between reference outputs and Fortran inference output is smaller than threshold, FAIL otherwise