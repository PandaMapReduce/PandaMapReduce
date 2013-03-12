# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#    NewPanda 						3/12/2013
#    							lihui@indiana.edu
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */




.KEEP_STATE:

all: mpi

OPTFLAGS    = -O 
INCFLAGS    = -I. -I/N/soft/intel/impi/4.0.0.028/include64/
CFLAGS      = $(OPTFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=0 
NVCCFLAGS   = $(CFLAGS) --ptxas-options=-v
LDFLAGS     = $(OPTFLAGS) -L/N/soft/intel/impi/4.0.0.028/lib64/
#LIBS        = -lmpi -lcudart -fopenmp -lcutil_x86_64
LIBS        = -lmpi -fopenmp 


#  note:
#  gcc usess -fopenmp
#  icc uses -opnemp

OMPFLAGS    = -fopenmp

CC          = g++
MPICC       = g++
NVCC        = nvcc

.c.o:
	$(CC) $(CFLAGS) -c $<

H_FILES     = kmeans.h
NVCCFLAGS  += -Xcompiler -fopenmp
INCFLAGS   += -I/opt/cuda-5.0/include/ -I/opt/NVIDIA_GPU_Computing_SDK/C/common/inc/
LDFLAGS    += -L/opt/cuda-5.0/lib64/ -L/opt/NVIDIA_GPU_Computing_SDK/C/lib/



MPI_SRC     = Chunk.cpp Mapper.cpp PandaChunk.cpp PartialReducer.cpp Reducer.cpp Combiner.cpp \
		MapReduceJob.cpp PandaMapReduceJob.cpp PartialSorter.cpp SerializedItemCollection.cpp \
		EmitConfiguration.cpp Message.cpp PandaMessage.cpp Partitioner.cpp Sorter.cpp \
		ItemCollection.cpp PandaMessageIORequest.cpp PreLoadedPandaChunk.cpp main.cpp \

CUDACPP_SRC = DeviceProperties.cpp  myString.cpp  Runtime.cpp  Stream.cpp
MPI_CUDA_SRC	= Panda2Lib.cu PandaSched.cu PandaUtils.cu UserAPI.cu PandaSort.cu PandaLib.cu

MPI_C_OBJ     	= $(MPI_SRC:%.c=%.o)
MPI_CUDA_OBJ	= $(MPI_CUDA_SRC:%.cu=%.o)

mpi_main.o: mpi_main.c $(H_FILES)
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -c $*.c

mpi_openmp_kmeans.o: mpi_openmp_kmeans.c $(H_FILES)
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -c $*.c

mpi_io.o: mpi_io.c $(H_FILES)
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -c $*.c

mpi_cuda_kmeans.o: mpi_cuda_kmeans.cu $(H_FILES)
	$(NVCC) $(LDFLAGS) -c $*.cu $(INCFLAGS) $(NVCCFLAGS) -lcudart -lmpi -lcutil_x86_64

mpi: mpi_main
mpi_main: $(MPI_C_OBJ) $(MPI_CUDA_OBJ) $(H_FILES)
	$(MPICC) $(LDFLAGS) $(OMPFLAGS) -o mpi_main $(MPI_C_OBJ) $(MPI_CUDA_OBJ) $(LIBS)


clean:
	rm -rf *.o mpi_main  \
	       core* .make.state gmon.out     \
               *.cluster_centres *.membership 
