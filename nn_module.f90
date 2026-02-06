! Steps to read the architecture and weights for a neural network and do inference on test data.

MODULE nn_module
    IMPLICIT NONE
    PRIVATE
    PUBLIC :: normalize_input, read_nn_architecture, read_nn_weights, read_nn_biases, read_nn_activation, forward_pass_single, forward_pass_full

    CONTAINS

        SUBROUTINE normalize_input(var_name, input, normalized_input)
            ! Normalizes the input using the mean and std from ./normalization_params. This is a common preprocessing step for neural network inputs.
            ! Needs to be called for every input before passing to the network!
            CHARACTER(LEN=*), INTENT(IN) :: var_name
            REAL, INTENT(IN) :: input
            REAL :: mean
            REAL :: std
            REAL, INTENT(OUT) :: normalized_input

            CHARACTER(LEN=32) :: mean_fname, std_fname
            INTEGER :: iunit

            WRITE(mean_fname, '(A,A,A)') './normalization_params/mean/', var_name, '.bin'
            OPEN(NEWUNIT=iunit, FILE=TRIM(mean_fname), STATUS='old', ACTION='read', FORM='unformatted', ACCESS='stream')
            READ(iunit) mean
            CLOSE(iunit)

            WRITE(std_fname, '(A,A,A)') './normalization_params/std/', var_name, '.bin'
            OPEN(NEWUNIT=iunit, FILE=TRIM(std_fname), STATUS='old', ACTION='read', FORM='unformatted', ACCESS='stream')
            READ(iunit) std
            CLOSE(iunit)

            normalized_input = (input - mean) / std

        END SUBROUTINE normalize_input

        SUBROUTINE read_nn_architecture(nlname, nnname, nl, layer_sizes)
            CHARACTER(LEN=32) :: nlname, nnname
            INTEGER, INTENT(OUT) :: nl
            INTEGER, ALLOCATABLE, INTENT(OUT) :: layer_sizes(:)
            INTEGER :: iunit

            WRITE(nlname, '(A)') './weights/nlayers.bin'
            OPEN(NEWUNIT=iunit, FILE=TRIM(nlname), STATUS='old', ACTION='read', FORM='unformatted', ACCESS='stream')
            READ(iunit) nl
            CLOSE(iunit)

            ALLOCATE(layer_sizes(nl+1)) ! one layer is (nnneurons0, nneurons1), so we need nlayers+1 entries

            WRITE(nnname, '(A)') './weights/nneurons.bin'
            OPEN(NEWUNIT=iunit, FILE=TRIM(nnname), STATUS='old', ACTION='read', FORM='unformatted', ACCESS='stream')
            READ(iunit) layer_sizes
            CLOSE(iunit)
        END SUBROUTINE read_nn_architecture

        SUBROUTINE read_nn_weights(filename_weights, weights)
            ! Can read weights
            CHARACTER(LEN=*), INTENT(IN) :: filename_weights
            REAL, INTENT(OUT) :: weights(:,:) ! Dimensions: input neurons, output neurons
            INTEGER :: iunit

            OPEN(NEWUNIT=iunit, FILE=filename_weights, STATUS='old', ACTION='read', ACCESS='stream', FORM='unformatted')
            READ(iunit) weights
            CLOSE(iunit)

        END SUBROUTINE read_nn_weights

        SUBROUTINE read_nn_biases(filename_biases, biases)
            ! Can read biases
            CHARACTER(LEN=*), INTENT(IN) :: filename_biases
            REAL, INTENT(OUT) :: biases(:) ! Dimensions: number of input neurons
            INTEGER :: iunit

            OPEN(NEWUNIT=iunit, FILE=filename_biases, STATUS='old', ACTION='read', ACCESS='stream', FORM='unformatted')
            READ(iunit) biases
            CLOSE(iunit)

        END SUBROUTINE read_nn_biases

        SUBROUTINE read_nn_activation(filename, activation)
            ! Reads activations. Routine is called layerwise, so this is only for one layer. (single string)
            CHARACTER(LEN=*), INTENT(IN) :: filename
            CHARACTER(LEN=*), INTENT(OUT) :: activation
            INTEGER :: iunit

            OPEN(NEWUNIT=iunit, FILE=filename, STATUS='old', ACTION='read')
            READ(iunit, *) activation
            CLOSE(iunit)

        END SUBROUTINE read_nn_activation

        SUBROUTINE forward_pass_single(inputs, weights, biases, activation, results)
            ! Does the inference step for a single layer
            REAL, INTENT(IN) :: weights(:,:)
            REAL, INTENT(IN) :: biases(:)
            CHARACTER(LEN=16), INTENT(IN) :: activation
            REAL, INTENT(IN) :: inputs(:)
            REAL, ALLOCATABLE, INTENT(OUT) :: results(:)

            ! Iterate through layers (infer from size of weights)
            ! Matrix multiplication at each layer
            ALLOCATE(results(size(weights(2,:))))
            results = matmul(inputs, weights) + biases
            IF (activation == 'relu') THEN
                ! write(*,*) 'Applying relu activation'
                results = max(results, 0.0)
            ELSEIF (activation == 'id') THEN
                ! write(*,*) 'No activation (identity)'
                ! Do nothing
            ELSE
                ! write(*,*) 'Unknown activation function:', activation
            END IF

        END SUBROUTINE forward_pass_single

        SUBROUTINE forward_pass_full(input, nlayers, layer_sizes, output)
            ! Does the full forward pass through the network
            REAL, INTENT(IN) :: input(:)
            INTEGER, INTENT(IN) :: nlayers
            INTEGER, INTENT(IN) :: layer_sizes(:)
            REAL, ALLOCATABLE, INTENT(OUT) :: output(:)
            INTEGER :: n, u
            CHARACTER(LEN=64) :: wname, bname, aname
            REAL, ALLOCATABLE :: current_input(:), current_output(:)
            REAL, ALLOCATABLE :: current_weights(:,:), current_biases(:)
            CHARACTER(LEN=16) :: current_activation

            current_input = input
            
            DO n=0, nlayers-1
                ! write(*,*) 'Processing layer ', n
                ! Remove allocation from possible previous iteration
                IF (allocated(current_weights)) DEALLOCATE(current_weights)
                IF (allocated(current_biases)) DEALLOCATE(current_biases)

                ! Read weights, biases, activations for layer n
                ! write(*,*) 'Reading weights...'
                WRITE(wname, '(A,I0,A)') 'weights/layer_', n, '_weights.bin'
                ALLOCATE(current_weights(layer_sizes(n+1), layer_sizes(n+2)))
                CALL read_nn_weights(wname, current_weights)
                ! write(*,*) 'shape of current_weights after reading: ', shape(current_weights)
                
                ! write(*,*) 'Reading biases...'
                WRITE(bname, '(A,I0,A)') 'weights/layer_', n, '_biases.bin'
                ALLOCATE(current_biases(layer_sizes(n+2)))
                CALL read_nn_biases(bname, current_biases)
                
                ! write(*,*) 'Reading activation...'
                WRITE(aname, '(A,I0,A)') 'weights/layer_', n, '_act.txt'
                CALL read_nn_activation(aname, current_activation)

                ! write(*,*) 'Layer ', n, ': weights shape = ', shape(current_weights), ', biases shape = ', shape(current_biases)

                CALL forward_pass_single(current_input, current_weights, current_biases, current_activation, current_output)

                current_input = current_output

            END DO

            output = current_output

        END SUBROUTINE forward_pass_full

END MODULE nn_module