#CC=gcc
CC=${CK_CC} 
LD=$(CC)
EXEC=${CK_PROG_TARGET_EXE}
#CUDAHOME=/usr/local/cuda-8.0
CUDAHOME=${CK_ENV_COMPILER_CUDA}
CUDACC=$(CUDAHOME)/bin/nvcc 
CUDA_ARCH=-arch=sm_30
CUDACFLAGS=-m64 -c -O3  --ptxas-options=-v -DCUDA 

CFLAGS=-g -W -Wall -Wno-unused-function -Wno-unused-parameter -c -O3  -DCUDA   -I$(CUDAHOME)/include
LDFLAGS =  -L$(CUDAHOME)/lib64 -lcudart -lstdc++ -lm # -lnvToolsExt




OBJ=bc2d.o cuda_kernels.o 


bc2d: ${OBJ}
	${LD} -o $(EXEC) ${OBJ} ${LDFLAGS}

.c.o:; $(CC) $(CFLAGS)     $< -o $@

%.o: %.cu
	$(CUDACC) $(CUDACFLAGS) $(CUDA_ARCH) -I$(MPI_DIR)/include -lm $<
clean:
	        rm -rf *.o $(EXEC)
