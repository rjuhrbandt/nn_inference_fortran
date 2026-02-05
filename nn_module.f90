! Steps to read the architecture and weights for a neural network and do inference on test data.

module nn_module
    implicit none
    private
    public :: read_nn_architecture, read_nn_weights, read_nn_biases, read_nn_activation, forward_pass_single, forward_pass_full

    contains

        subroutine read_nn_architecture(nlname, nnname, nl, layer_sizes)
            character(len=32) :: nlname, nnname
            integer, intent(out) :: nl
            integer, allocatable, intent(out) :: layer_sizes(:)
            integer :: iunit

            write(nlname, '(A)') 'weights/nlayers.bin'
            open(newunit=iunit, file=trim(nlname), status='old', action='read', form='unformatted', access='stream')
            read(iunit) nl
            close(iunit)

            allocate(layer_sizes(nl+1)) ! one layer is (nnneurons0, nneurons1), so we need nlayers+1 entries

            write(nnname, '(A)') 'weights/nneurons.bin'
            open(newunit=iunit, file=trim(nnname), status='old', action='read', form='unformatted', access='stream')
            read(iunit) layer_sizes
            close(iunit)
        end subroutine read_nn_architecture

        subroutine read_nn_weights(filename_weights, weights)
            ! Can read weights
            character(len=*), intent(in) :: filename_weights
            real, intent(out) :: weights(:,:) ! Dimensions: input neurons, output neurons
            integer :: iunit

            open(newunit=iunit, file=filename_weights, status='old', access='stream', form='unformatted')
            read(iunit) weights
            close(iunit)

        end subroutine read_nn_weights

        subroutine read_nn_biases(filename_biases, biases)
            ! Can read biases
            character(len=*), intent(in) :: filename_biases
            real, intent(out) :: biases(:) ! Dimensions: number of input neurons
            integer :: iunit

            open(newunit=iunit, file=filename_biases, status='old', action='read', access='stream', form='unformatted')
            read(iunit) biases
            close(iunit)

        end subroutine read_nn_biases

        subroutine read_nn_activation(filename, activation)
            ! Reads activations. Routine is called layerwise, so this is only for one layer. (single string)
            character(len=*), intent(in) :: filename
            character(len=*), intent(out) :: activation
            integer :: iunit

            open(newunit=iunit, file=filename, status='old', action='read')
            read(iunit, *) activation
            close(iunit)

        end subroutine read_nn_activation

        subroutine forward_pass_single(inputs, weights, biases, activation, results)
            ! Does the inference step for a single layer
            real, intent(in) :: weights(:,:)
            real, intent(in) :: biases(:)
            character(len=16), intent(in) :: activation
            real, intent(in) :: inputs(:)
            real, allocatable, intent(out) :: results(:)

            ! Iterate through layers (infer from size of weights)
            ! Matrix multiplication at each layer
            allocate(results(size(weights(2,:))))
            results = matmul(inputs, weights) + biases
            IF (activation == 'relu') THEN
                write(*,*) 'Applying relu activation'
                results = max(results, 0.0)
            ELSEIF (activation == 'id') THEN
                write(*,*) 'No activation (identity)'
                ! Do nothing
            ELSE
                write(*,*) 'Unknown activation function:', activation
            END IF

        end subroutine forward_pass_single

        subroutine forward_pass_full(input, nlayers, layer_sizes, output)
            ! Does the full forward pass through the network
            real, intent(in) :: input(:)
            integer, intent(in) :: nlayers
            integer, intent(in) :: layer_sizes(:)
            real, allocatable, intent(out) :: output(:)
            integer :: n, u
            character(len=64) :: wname, bname, aname
            real, allocatable :: current_input(:), current_output(:)
            real, allocatable :: current_weights(:,:), current_biases(:)
            character(len=16) :: current_activation

            current_input = input
            
            DO n=0, nlayers-1
                write(*,*) 'Processing layer ', n
                ! Remove allocation from possible previous iteration
                if (allocated(current_weights)) deallocate(current_weights)
                if (allocated(current_biases)) deallocate(current_biases)

                ! Read weights, biases, activations for layer n
                write(*,*) 'Reading weights...'
                write(wname, '(A,I0,A)') 'weights/layer_', n, '_weights.bin'
                allocate(current_weights(layer_sizes(n+1), layer_sizes(n+2)))
                write(*,*) 'shape of current_weights: ', shape(current_weights)
                call read_nn_weights(wname, current_weights)
                write(*,*) 'shape of current_weights after reading: ', shape(current_weights)
                
                write(*,*) 'Reading biases...'
                write(bname, '(A,I0,A)') 'weights/layer_', n, '_biases.bin'
                allocate(current_biases(layer_sizes(n+2)))
                call read_nn_biases(bname, current_biases)
                
                write(*,*) 'Reading activation...'
                write(aname, '(A,I0,A)') 'weights/layer_', n, '_act.txt'
                call read_nn_activation(aname, current_activation)

                write(*,*) 'Layer ', n, ': weights shape = ', shape(current_weights), ', biases shape = ', shape(current_biases)

                call forward_pass_single(current_input, current_weights, current_biases, current_activation, current_output)

                current_input = current_output

            END DO

            output = current_output

        end subroutine forward_pass_full

end module nn_module