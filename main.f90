! This is the "testing environment" Fortran program for validating the
! Fortran neural network implementation against reference outputs from a
! PyTorch model.

! Steps:
! 1. Load the neural network weights and biases from the binary files in ./weights
! 2. Load test inputs and reference outputs from ./test_io
! 3. For each test case, run the Fortran forward pass and compare outputs
! 4. Report PASS/FAIL for each test case and overall summary: PASS IF dIFference between Fortran and Python outputs is < 1e-5, ELSE FAIL

PROGRAM nn_inference
  USE nn_module
  IMPLICIT NONE

  INTEGER :: nlayers, u, i
  REAL, ALLOCATABLE :: input(:), output(:), ref_output(:)
  REAL :: max_err
  INTEGER :: num_tests
  CHARACTER(LEN=100) :: fname
  CHARACTER(LEN=32) :: nlname='weights/nlayers.bin', nnname='weights/nneurons.bin'
  INTEGER, ALLOCATABLE :: layer_sizes(:)

  ! Step 1: READ architecture to determine sizes
  ! WRITE(*,*) 'READing neural network architecture...'
  CALL READ_nn_architecture(nlname, nnname, nlayers, layer_sizes)
  ! WRITE(*,*) 'Found nlayers: ', nlayers
  ! WRITE(*,*) 'Found layer sizes: ', layer_sizes

  ! Iterate through num_tests test cases
  WRITE(fname, '(A)') 'test_io/num_tests.bin'
  OPEN(newunit=u, file=trim(fname), status='old', access='stream', form='unformatted', action='READ')
  READ(u) num_tests
  CLOSE(u)
  ! WRITE(*,*) 'Number of test cases: ', num_tests

  ! ALLOCATE array for inputs and outputs
  ALLOCATE(input(layer_sizes(1)))
  ALLOCATE(ref_output(layer_sizes(nlayers+1)))

  DO i=0, num_tests-1
    WRITE(*,*) 'Running test case ', i

    ! Step 2: READ inputs and outputs
    ! WRITE(*,*) 'READing input array...'
    WRITE(fname, '(A,I0,A)') 'test_io/input_', i, '.bin'
    OPEN(newunit=u, file=trim(fname), access='stream', form='unformatted', status='old')
    READ(u) input
    CLOSE(u)
    ! WRITE(*,*) 'READing output array...'
    WRITE(fname, '(A,I0,A)') 'test_io/output_', i, '.bin'
    OPEN(newunit=u, file=trim(fname), access='stream', form='unformatted', status='old')
    READ(u) ref_output
    CLOSE(u)

    ! Step 3: Perform the forward pass for each layer
    CALL forward_pass_full(input, nlayers, layer_sizes, output)
    
    ! Compare outputs and report PASS/FAIL
    ! WRITE(*,*) 'Comparing outputs...'
    max_err = maxval(abs(output - ref_output))
    WRITE(*,*) 'Output from inference: ', output
    WRITE(*,*) 'Reference output:     ', ref_output
    WRITE(*,*) 'Maximum absolute error: ', max_err
    IF (max_err < 1e-5) THEN
      ! WRITE(*, '(A,I0,A,F10.6)') 'Test case: PASS (max error = ', max_err, ')'
      WRITE(*,*) 'Test case: PASS'
    ELSE
      ! WRITE(*, '(A,I0,A,F10.6)') 'Test case: FAIL (max error = ', max_err, ')'
      WRITE(*,*) 'Test case: FAIL'
    END IF

  END DO
    
  DEALLOCATE(input)
  DEALLOCATE(ref_output)
  DEALLOCATE(layer_sizes)

END PROGRAM nn_inference