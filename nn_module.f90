! Steps to read the architecture and weights for a neural network and do inference on test data.

module nn_module
    implicit none
    private
    public :: read_nn_architecture, read_nn_weights, read_nn_biases, read_nn_activation, perform_nn_inference

    contains

        subroutine read_nn_architecture(filename, nlayers, layer_sizes)
            character(len=*), intent(in) :: filename
            integer, intent(out) :: nlayers
            integer, allocatable, intent(out) :: layer_sizes(:)
            integer :: iunit, i

            open(newunit=iunit, file=filename, status='old', action='read')
            read(iunit, *) nlayers
            allocate(layer_sizes(nlayers))
            do i = 1, nlayers
                read(iunit, *) layer_sizes(i)
            end do
            close(iunit)
        end subroutine read_nn_architecture

        subroutine read_nn_weights(filename_weights, weights)
            ! Can read weights
            character(len=*), intent(in) :: filename_weights
            real, intent(out) :: weights(:,:) ! Dimensions: layer index, input neurons, output neurons
            integer :: iunit

            open(newunit=iunit, file=filename_weights, status='old', action='read')
            read(iunit, *) weights
            close(iunit)

        end subroutine read_nn_weights

        subroutine read_nn_biases(filename_biases, biases)
            ! Can read biases
            character(len=*), intent(in) :: filename_biases
            real, intent(out) :: biases(:) ! Dimensions: layer index, number of input neurons
            integer :: iunit

            open(newunit=iunit, file=filename_biases, status='old', action='read')
            read(iunit, *) biases
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

        subroutine perform_nn_inference(inputs, weights, biases, activations, outputs)
            ! Does the inference step
            real, intent(in) :: weights(:,:,:)
            real, intent(in) :: biases(:,:)
            character(len=16), intent(in) :: activations(:)
            integer :: nlayers
            real :: inputs(:)
            real, allocatable, intent(out) :: outputs(:)
            integer :: nl
            integer :: ninputs ! Length of input array, for catching errors
            real, allocatable :: current_weights(:,:), current_biases(:), result(:)
            character(len=16) :: current_activation
            
            nlayers = size(weights, 1)

            ! Read input values
            ! Check if they have the same length as the second dimension of weights: size(weights(1), 1)
            ninputs = size(inputs)
            IF (ninputs /= size(weights(1,:,:), 1)) THEN
                write(*,*) 'Input size not compatible with size of network!'
            END IF

            allocate(current_weights(size(weights, 2), size(weights, 3)))
            allocate(current_biases(size(biases, 2)))

            ! Iterate through layers (infer from size of weights)
            ! Matrix multiplication at each layer
            DO nl = 1, nlayers
                write(*,*) 'Doing operations for layer', nl
                current_weights = weights(nl,:,:)
                current_biases = biases(nl,:)
                current_activation = activations(nl)
                allocate(result(size(current_weights(1,:))))
                result = matmul(inputs, current_weights)
                result = result + current_biases
                IF (current_activation == 'relu') THEN
                    write(*,*) 'Applying relu activation at layer', nl
                    result = max(result, 0.0)
                ELSEIF (current_activation == 'id') THEN
                    write(*,*) 'No activation (identity) at layer', nl
                    ! Do nothing
                ELSE
                    write(*,*) 'Unknown activation function:', current_activation
                END IF
                ! Assign value of this layer's result to input for next iteration
                inputs = result
            END DO
            
            outputs = inputs
            deallocate(result)
            deallocate(current_weights)
            deallocate(current_biases)

        end subroutine perform_nn_inference

end module nn_module