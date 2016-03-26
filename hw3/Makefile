NVCC      = nvcc
CFLAGS    = -O3
LDFLAGS   =

OBJ     = time_it.o
HDRS    = time_it.h

perc: perc.o $(OBJ)
	$(NVCC) $(CFLAGS) perc.o $(OBJ) $(LDFLAGS) -o perc

examples: examples.o
	$(NVCC) $(CFLAGS) examples.o $(LDFLAGS) -o examples

hw3tests: hw3tests.o
	$(NVCC) $(CFLAGS) hw3tests.o $(LDFLAGS) -o hw3tests

timing: timing.o $(OBJ)
	$(NVCC) $(CFLAGS) timing.o $(OBJ) $(LDFLAGS) -o timing

hw3: hw3.o $(OBJ)
	$(NVCC) $(CFLAGS) hw3.o $(OBJ) $(LDFLAGS) -o hw3

clean:
	$(RM) *.o perc examples timing hw3tests hw3


.SUFFIXES: .c .cu .o
.c.o: $(HDRS)
	$(NVCC) -c $(CFLAGS) $<
.cu.o: $(HDRS)
	$(NVCC) -c $(CFLAGS) $<
