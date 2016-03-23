NVCC      = nvcc
CFLAGS    = -O3
LDFLAGS   =

OBJ     = time_it.o
HDRS    = time_it.h

perc: perc.o $(OBJ)
	$(NVCC) $(CFLAGS) perc.o $(OBJ) $(LDFLAGS) -o perc

examples: examples.o
	$(NVCC) $(CFLAGS) examples.o $(LDFLAGS) -o examples

hw3: hw3.o
	$(NVCC) $(CFLAGS) hw3.o $(LDFLAGS) -o hw3

timing: timing.o $(OBJ)
	$(NVCC) $(CFLAGS) timing.o $(OBJ) $(LDFLAGS) -o timing

hw3timing: hw3timing.o $(OBJ)
	$(NVCC) $(CFLAGS) hw3timing.o $(OBJ) $(LDFLAGS) -o hw3timing

clean:
	$(RM) *.o perc examples timing hw3 hw3timing


.SUFFIXES: .c .cu .o
.c.o: $(HDRS)
	$(NVCC) -c $(CFLAGS) $<
.cu.o: $(HDRS)
	$(NVCC) -c $(CFLAGS) $<
