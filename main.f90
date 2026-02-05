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

  integer :: i, u, nlayers_max=10
  real, allocatable :: weights(:,:,:), biases(:,:), input(:), output(:), ref_output(:)
  real :: max_err
  integer :: max_num_neurons = 1024
  integer :: num_tests
  character(len=100) :: fname, wname, bname, aname
  character(len=16), allocatable :: activations(:) ! Activations do not have long names

  ! Step 1: Load weights, biases and activations from ./weights
    allocate(weights(nlayers_max,max_num_neurons,max_num_neurons))
    allocate(biases(nlayers_max,max_num_neurons))
    allocate(activations(nlayers_max))

    weights = 0.0
    biases = 0.0
    activations = 'id' ! Default activation is identity (for safety)

  ! Do something until an error occurs, then break/continue
    DO i = 1, nlayers_max  ! Hard-coded
        ! Load weights and biases for layer i
        
        write(wname, '(A,I0,A)') 'weights/layer_', i-1, '_weights.bin'
        call read_nn_weights(wname, weights(i,:,:))
        write(bname, '(A,I0,A)') 'weights/layer_', i-1, '_biases.bin'
        call read_nn_biases(bname, biases(i,:))
        write(aname, '(A,I0,A)') 'weights/layer_', i-1, '_act.txt'
        call read_nn_activation(aname, activations(i))
        ! If error occurs (e.g., file not found), exit loop
        EXIT
    END DO
    
    ! Step 2: Load test inputs and reference outputs from ./test_io
    ! How many test cases?
    open(unit=u, file='test_io/num_tests.txt', status='old')
    read(u, *) num_tests
    close(u)

    DO i = 0, num_tests - 1
        write(*, '(A,I0,A)') 'Running test case ', i, '...'
        ! Read test input
        write(fname, '(A,I0,A)') 'test_io/input_', i, '.bin'
        open(newunit=u, file=trim(fname), access='stream', form='unformatted', status='old')
        read(u) input
        close(u)

        ! Read reference output
        write(fname, '(A,I0,A)') 'test_io/output_', i, '.bin'
        open(newunit=u, file=trim(fname), access='stream', form='unformatted', status='old')
        read(u) ref_output
        close(u)

        ! Run Fortran forward pass
        call perform_nn_inference(input, weights, biases, activations, output)

        ! Compare outputs and report PASS/FAIL
        max_err = maxval(abs(output - ref_output))
        if (max_err < 1e-5) then
            write(*, '(A,I0,A,F10.6)') 'Test case ', i, ': PASS (max error = ', max_err, ')'
        else
            write(*, '(A,I0,A,F10.6)') 'Test case ', i, ': FAIL (max error = ', max_err, ')'
        end if
    END DO

    deallocate(weights)
    deallocate(biases)
    deallocate(activations)

END PROGRAM nn_inference