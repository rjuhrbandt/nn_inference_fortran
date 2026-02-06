! This is the "testing environment" Fortran program for validating the
! Fortran neural network implementation against reference outputs from a
! PyTorch model.

! Steps:
! 1. Load the neural network weights and biases from the binary files in ./weights
! 2. Load test inputs and reference outputs from ./test_io
! 3. For each test case, run the Fortran forward pass and compare outputs
! 4. Report PASS/FAIL for each test case and overall summary: PASS if difference between Fortran and Python outputs is < 1e-5, else FAIL

PROGRAM nn_inference
  use nn_module
  implicit none

  integer :: nlayers, u, i
  real, allocatable :: input(:), output(:), ref_output(:)
  real :: max_err
  integer :: num_tests
  character(len=100) :: fname
  character(len=32) :: nlname='weights/nlayers.bin', nnname='weights/nneurons.bin'
  integer, allocatable :: layer_sizes(:)

  ! Step 1: Read architecture to determine sizes
  ! write(*,*) 'Reading neural network architecture...'
  call read_nn_architecture(nlname, nnname, nlayers, layer_sizes)
  ! write(*,*) 'Found nlayers: ', nlayers
  ! write(*,*) 'Found layer sizes: ', layer_sizes

  ! Iterate through num_tests test cases
  write(fname, '(A)') 'test_io/num_tests.bin'
  open(newunit=u, file=trim(fname), status='old', access='stream', form='unformatted', action='read')
  read(u) num_tests
  close(u)
  ! write(*,*) 'Number of test cases: ', num_tests

  ! Allocate array for inputs and outputs
  allocate(input(layer_sizes(1)))
  allocate(ref_output(layer_sizes(nlayers+1)))

  DO i=0, num_tests-1
    write(*,*) 'Running test case ', i

    ! Step 2: Read inputs and outputs
    ! write(*,*) 'Reading input array...'
    write(fname, '(A,I0,A)') 'test_io/input_', i, '.bin'
    open(newunit=u, file=trim(fname), access='stream', form='unformatted', status='old')
    read(u) input
    close(u)
    ! write(*,*) 'Reading output array...'
    write(fname, '(A,I0,A)') 'test_io/output_', i, '.bin'
    open(newunit=u, file=trim(fname), access='stream', form='unformatted', status='old')
    read(u) ref_output
    close(u)

    ! Step 3: Perform the forward pass for each layer
    call forward_pass_full(input, nlayers, layer_sizes, output)
    
    ! Compare outputs and report PASS/FAIL
    ! write(*,*) 'Comparing outputs...'
    max_err = maxval(abs(output - ref_output))
    write(*,*) 'Output from inference: ', output
    write(*,*) 'Reference output:     ', ref_output
    write(*,*) 'Maximum absolute error: ', max_err
    if (max_err < 1e-5) then
      ! write(*, '(A,I0,A,F10.6)') 'Test case: PASS (max error = ', max_err, ')'
      write(*,*) 'Test case: PASS'
    else
      ! write(*, '(A,I0,A,F10.6)') 'Test case: FAIL (max error = ', max_err, ')'
      write(*,*) 'Test case: FAIL'
    end if

  END DO
    
  deallocate(input)
  deallocate(ref_output)
  deallocate(layer_sizes)

END PROGRAM nn_inference