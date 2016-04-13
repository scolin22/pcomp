NVCC      = nvcc
CFLAGS    = -O3 
LDFLAGS   =

OBJ     = time_it.o
HDRS    = time_it.h

perc0: perc0.o $(OBJ)
	$(NVCC) $(CFLAGS) perc0.o $(OBJ) $(LDFLAGS) -o perc0

perc: perc.o $(OBJ)
	$(NVCC) $(CFLAGS) perc.o $(OBJ) $(LDFLAGS) -o perc

.SUFFIXES: .c .cu .o
.c.o: $(HDRS)
	$(NVCC) -c $(CFLAGS) $<
.cu.o: $(HDRS)
	$(NVCC) -c $(CFLAGS) $<
