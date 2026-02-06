! Steps to read the architecture and weights for a neural network and do inference on test data.

MODULE nn_module
    IMPLICIT NONE
    PRIVATE
    PUBLIC :: normalize_input, read_nn_architecture, read_nn_weights, read_nn_biases, read_nn_activation 
    PUBLIC :: forward_pass_single, forward_pass_full

    CONTAINS

        SUBROUTINE assign_input_values(input_var_names, num_input_var_names, z, n, ordered_nb_list, input_data, n_input_data)
            ! This is a placeholder for the logic to extract the input data from the FESOM data structure based on the variable name. 
            ! This will likely involve a series of IF statements or a SELECT CASE statement to match the variable names to the corresponding data extraction logic.
            CHARACTER(LEN=*), INTENT(IN) :: input_var_names(:)
            REAL, INTENT(OUT) :: input_data(:), n_input_data(:) 
            INTEGER, INTENT(IN) :: z, n ! These are the vertical level and node index for which we want to extract the data. They are needed to know which value to extract from the FESOM data structure.
            INTEGER :: i ! Current index in loop over input variables. var_name and var_name_nb... are successive in the input_var_names.txt, so we can just keep incrementing j to fill the input_data array in the correct order.
            INTEGER :: nb, u
            INTEGER :: max_num_nb = 6 ! Maximum number of neighbours in dbgyre mesh.
            INTEGER :: num_input_var_names
            INTEGER, INTENT(IN) :: ordered_nb_list(:) ! This is the list of neighbour node indices, ordered ascending.
            CHARACTER(LEN=32) :: name, fname
            REAL, ALLOCATABLE :: means(:), stds(:) ! Arrays to hold the means and stds for all input variables, to be read from ./normalization_params. This is needed for normalization of the input data.

            ALLOCATE(means(num_input_var_names), stds(num_input_var_names))

            OPEN(NEWUNIT=u, FILE=TRIM(fname), STATUS='old', ACTION='read', FORM='unformatted', ACCESS='stream')
            DO i = 1, num_input_var_names
                 ! Loop over input variable names. For each variable name, extract the corresponding data from the FESOM data structure and fill the input_data array. 
                 ! Also apply normalization to the input data using the mean and std from the training data, which are stored in ./normalization_params.
                name = input_var_names(i)
                WRITE(fname, '(A,A,A)') './normalization_params/mean/', input_var_names(i), '.bin'
                READ(u) means(i)
                WRITE(fname, '(A,A,A)') './normalization_params/std/', input_var_names(i), '.bin'
                READ(u) stds(i)
                SELECT CASE (name)
                    CASE ('temp')
                        ! input_data(i) = tracers%data(1)%values(z,n)
                        DO nb=1, max_num_nb
                            IF ( ordered_nb_list(nb) == -1 ) THEN
                                input_data(i+nb) = means(i) ! or, equivalently, the mean value from the training data after normalization
                            ELSE
                                !  input_data(i+nb) = tracers%data(1)%values(z, ordered_nb_list(nb))
                            END IF
                        END DO
                    CASE ('unod')
                        ! input_data(i) = dynamics%uvnode(1,z,n)
                        DO nb=1, max_num_nb
                            IF ( ordered_nb_list(nb) == -1 ) THEN
                                input_data(i+nb) = means(i) ! or, equivalently, the mean value from the training data after normalization
                            ELSE
                                !  input_data(i+nb) = dynamics%uvnode(1,z, ordered_nb_list(nb))
                            END IF
                        END DO
                    CASE ('vnod')
                        ! input_data(i) = dynamics%uvnode(2,z,n)
                        DO nb=1, max_num_nb
                            IF ( ordered_nb_list(nb) == -1 ) THEN
                                input_data(i+nb) = means(i) ! or, equivalently, the mean value from the training data after normalization
                            ELSE
                                !  input_data(i+nb) = dynamics%uvnode(2,z, ordered_nb_list(nb))
                            END IF
                        END DO
                    CASE ('curl_u')
                        ! Call customized subroutine to compute curl_u as it is not computed by default
                        ! curl_u = ...
                        DO nb=1, max_num_nb
                            IF ( ordered_nb_list(nb) == -1 ) THEN
                                input_data(i+nb) = means(i) ! or, equivalently, the mean value from the training data after normalization
                            ELSE
                                !  input_data(i+nb) = curl_u(z, ordered_nb_list(nb))
                            END IF
                        END DO
                    CASE ('slope_x')
                        ! input_data(i) = neutral_slope(1,z,n)
                        DO nb=1, max_num_nb
                            IF ( ordered_nb_list(nb) == -1 ) THEN
                                input_data(i+nb) = means(i) ! or, equivalently, the mean value from the training data after normalization
                            ELSE
                                !  input_data(i+nb) = neutral_slope(1,z, ordered_nb_list(nb))
                            END IF
                        END DO
                    CASE ('slope_y')
                        ! input_data(i) = neutral_slope(2,z,n)
                        DO nb=1, max_num_nb
                            IF ( ordered_nb_list(nb) == -1 ) THEN
                                input_data(i+nb) = means(i) ! or, equivalently, the mean value from the training data after normalization
                            ELSE
                                !  input_data(i+nb) = neutral_slope(2,z, ordered_nb_list(nb))
                            END IF
                        END DO
                    CASE ('N2')
                        ! Watch out: N2 is defined at vertical interfaces, so it needs to be interpolated to the vertical levels of the nodes.
                        ! input_data(i) = bvfreq(z,n)
                        DO nb=1, max_num_nb
                            IF ( ordered_nb_list(nb) == -1 ) THEN
                                input_data(i+nb) = means(i) ! or, equivalently, the mean value from the training data after normalization
                            ELSE
                                !  input_data(i+nb) = bvfreq(z, ordered_nb_list(nb))
                            END IF
                        END DO
                    CASE ('ld_baroc1')
                        ! input_data(i) = rosb(n)
                    CASE DEFAULT
                        ! WRITE(*,(A,A,A)) 'This variable (' // var_name // ') probably ends with _nb and its value is already filled. Skipping...'
                END SELECT

            END DO
            CLOSE(u)
            ! Now normalize by doing array computation
            n_input_data = (input_data - means) / stds
            DEALLOCATE(means, stds)
        END SUBROUTINE assign_input_values

        SUBROUTINE normalize_input(var_name,input, normalized_input)
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