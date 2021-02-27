.SUFFIXES: .cpp .cu .o

NVCC = nvcc --ptxas-options=-v -Xcompiler "-O3" -Xcompiler "-ffast-math" -O3 -use_fast_math -I. -G

OBJ1 = test.o fmm.o cpukernel.o
OBJ2 = test.o fmm.o ssekernel.o
OBJ3 = test.o fmm.o gpukernel_p3.o
OBJ4 = test.o fmm.o gpukernel_p4.o
LIB = -lcudart

all:
	make cpu1
	./a.out
	make cpu2
	./a.out
	make gpu3
	./a.out
	make gpu4
	./a.out
cpu1: $(OBJ1)
	$(NVCC) $? $(LIB)
cpu2: $(OBJ2)
	$(NVCC) $? $(LIB)
gpu3: $(OBJ3)
	$(NVCC) $? $(LIB)
gpu4: $(OBJ4)
	$(NVCC) $? $(LIB)
clean:
	$(RM) *.o *.out

.cpp.o:
	$(NVCC) -c $< -o $@
.cu.o:
	$(NVCC) -c $< -o $@
