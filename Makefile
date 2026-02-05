FC = gfortran
FFLAGS = -O2

TARGET = nn_module
OBJS = nn_module.o main.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(FC) $(FFLAGS) -o $@ $^

nn_module.o: nn_module.f90
	$(FC) $(FFLAGS) -c $<

main.o: main.f90 nn_module.o
	$(FC) $(FFLAGS) -c $<

clean:
	rm -f $(TARGET) $(OBJS) *.mod

.PHONY: all clean
