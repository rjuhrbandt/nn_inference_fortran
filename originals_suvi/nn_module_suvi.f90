!> Neural network inference module for a 3-layer MLP.
!>
!> Implements the same architecture as the PyTorch model in
!> nn_inference_minimal.py:
!>
!>   Input (50) -> Linear(50,512) + ReLU
!>              -> Linear(512,512) + ReLU
!>              -> Linear(512,3)            (no activation)
!>
!> Weight storage convention
!> -------------------------
!> PyTorch stores weight matrices as (out_features, in_features) in
!> row-major order.  For Fortran's column-major layout the export script
!> (export_weights.py) transposes and writes them in Fortran order, so
!> each weight array here is declared as w(in_features, out_features).
!>
!> The forward pass therefore computes each layer as:
!>
!>   output = matmul(input, W) + bias
!>
!> which is equivalent to PyTorch's  output = input @ W^T + bias  but
!> avoids an explicit transpose at runtime.
!>
!> Binary file format
!> ------------------
!> All .bin files are raw streams of IEEE 754 single-precision (float32)
!> values with no headers or record markers, read via Fortran stream I/O.
module nn_module
  implicit none

  integer, parameter :: INPUT_DIM  = 50   !< Number of input features
  integer, parameter :: HIDDEN_DIM = 512  !< Width of both hidden layers
  integer, parameter :: OUTPUT_DIM = 3    !< Number of output values

  !> Weight matrices stored as (in_features, out_features)
  real(4) :: w0(INPUT_DIM,  HIDDEN_DIM), b0(HIDDEN_DIM)
  real(4) :: w1(HIDDEN_DIM, HIDDEN_DIM), b1(HIDDEN_DIM)
  real(4) :: w2(HIDDEN_DIM, OUTPUT_DIM), b2(OUTPUT_DIM)

contains

  !> Load all weight and bias arrays from binary files.
  !>
  !> Expects the following files inside `dir`:
  !>   weight_0.bin, bias_0.bin   (layer 0: 50  -> 512)
  !>   weight_1.bin, bias_1.bin   (layer 1: 512 -> 512)
  !>   weight_2.bin, bias_2.bin   (layer 2: 512 -> 3)
  subroutine load_weights(dir)
    character(len=*), intent(in) :: dir

    call read_bin(trim(dir) // '/weight_0.bin', w0, size(w0))
    call read_bin(trim(dir) // '/bias_0.bin',   b0, size(b0))
    call read_bin(trim(dir) // '/weight_1.bin', w1, size(w1))
    call read_bin(trim(dir) // '/bias_1.bin',   b1, size(b1))
    call read_bin(trim(dir) // '/weight_2.bin', w2, size(w2))
    call read_bin(trim(dir) // '/bias_2.bin',   b2, size(b2))
  end subroutine

  !> Read `n` float32 values from a raw binary file into `arr`.
  subroutine read_bin(fname, arr, n)
    character(len=*), intent(in) :: fname
    integer, intent(in) :: n
    real(4), intent(out) :: arr(n)
    integer :: u

    open(newunit=u, file=fname, access='stream', form='unformatted', status='old')
    read(u) arr
    close(u)
  end subroutine

  !> Run a forward pass through the 3-layer MLP.
  !>
  !> Computation for a single input vector:
  !>   h1 = ReLU( matmul(input, w0) + b0 )    -- hidden layer 0
  !>   h2 = ReLU( matmul(h1,    w1) + b1 )    -- hidden layer 1
  !>   output =    matmul(h2,    w2) + b2      -- output layer (linear)
  !>
  !> ReLU is applied element-wise: max(x, 0).
  subroutine forward(input, output)
    real(4), intent(in)  :: input(INPUT_DIM)
    real(4), intent(out) :: output(OUTPUT_DIM)
    real(4) :: h1(HIDDEN_DIM), h2(HIDDEN_DIM)

    ! Layer 0: linear + ReLU
    h1 = matmul(input, w0) + b0
    h1 = max(h1, 0.0)

    ! Layer 1: linear + ReLU
    h2 = matmul(h1, w1) + b1
    h2 = max(h2, 0.0)

    ! Layer 2: linear (no activation)
    output = matmul(h2, w2) + b2
  end subroutine

end module nn_module
