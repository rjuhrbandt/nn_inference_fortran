!> Cross-validation program: Fortran inference vs Python reference.
!>
!> This program validates that the Fortran implementation in nn_module.f90
!> produces the same results as the PyTorch model.  It works as follows:
!>
!>   1. Load the neural network weights from weights/*.bin
!>      (produced by export_weights.py).
!>
!>   2. Read the number of test cases from weights/num_tests.txt.
!>
!>   3. For each test case i = 0 .. num_tests-1:
!>      a. Read the input vector  from weights/test_input_<i>.bin   (50 floats)
!>      b. Read the Python output from weights/test_output_<i>.bin  (3 floats)
!>      c. Run the Fortran forward pass to get Fortran output       (3 floats)
!>      d. Compute the maximum absolute error across the 3 outputs
!>      e. Report PASS if max error < 1e-5, otherwise FAIL
!>
!>   4. Print overall PASS/FAIL summary.  Exit code 1 if any test failed.
!>
!> The test vectors include both random inputs (with fixed seeds for
!> reproducibility) and edge cases (all-zeros, all-ones, all-negative),
!> covering a range of activation patterns in the hidden layers.
program nn_inference
  use nn_module
  implicit none

  integer :: num_tests, i, u
  real(4) :: input(INPUT_DIM), output(OUTPUT_DIM), ref_output(OUTPUT_DIM)
  real(4) :: max_err
  character(len=256) :: fname
  logical :: all_pass

  ! Load weights from binary files produced by export_weights.py
  call load_weights('weights')

  ! Read number of test cases
  open(newunit=u, file='weights/num_tests.txt', status='old')
  read(u, *) num_tests
  close(u)

  all_pass = .true.

  write(*,'(A,I0,A)') 'Running ', num_tests, ' test cases:'
  write(*,'(A)') repeat('-', 70)

  do i = 0, num_tests - 1
    ! Read test input (50 float32 values)
    write(fname, '(A,I0,A)') 'weights/test_input_', i, '.bin'
    open(newunit=u, file=trim(fname), access='stream', form='unformatted', status='old')
    read(u) input
    close(u)

    ! Read Python reference output (3 float32 values)
    write(fname, '(A,I0,A)') 'weights/test_output_', i, '.bin'
    open(newunit=u, file=trim(fname), access='stream', form='unformatted', status='old')
    read(u) ref_output
    close(u)

    ! Run Fortran forward pass
    call forward(input, output)

    ! Element-wise comparison
    max_err = maxval(abs(output - ref_output))

    write(*,'(A,I0)') 'Test case ', i
    write(*,'(A,3F14.8)') '  Python ref: ', ref_output
    write(*,'(A,3F14.8)') '  Fortran:    ', output
    write(*,'(A,ES12.5)') '  Max error:  ', max_err

    if (max_err < 1.0e-5) then
      write(*,'(A)') '  Result:     PASS'
    else
      write(*,'(A)') '  Result:     FAIL'
      all_pass = .false.
    end if
    write(*,'(A)') ''
  end do

  write(*,'(A)') repeat('=', 70)
  if (all_pass) then
    write(*,'(A)') 'ALL TESTS PASSED'
  else
    write(*,'(A)') 'SOME TESTS FAILED'
    stop 1
  end if

end program nn_inference
