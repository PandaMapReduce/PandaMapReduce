################################################################################
# Build script for OpenMP implementation of cmeans using multi-core on Delta
# lihui@indiana.edu   last update 5/23/2012
################################################################################

all:da_cmeans

CC	= g++

.c.o:
	$(CC) -c $<
OBJ		= Volume.o DACmeansMultiCPU.o MDL.o
INCLUDE   = -I. -I/N/soft/intel/impi/4.0.0.028/include64/
LIBPATH = -L/N/soft/intel/impi/4.0.0.028/lib64/

MDL.o: MDL.cpp
	$(CC) -c MDL.cpp
Volume.o: Volume.cpp
	$(CC) -c Volume.cpp 
DACmeansMultiCPU.o: DACmeansMultiCPU.cpp
	$(CC) -c DACmeansMultiCPU.cpp -o DACmeansMultiCPU.o -fopenmp $(LIBPATH) $(INCLUDE)

da_cmeans: $(OBJ)
	$(CC) -o da_cmeans $(OBJ) -fopenmp -lmpi $(LIBPATH)
clean:
	rm *.o da_cmeans
