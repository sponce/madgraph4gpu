FLAGS = -O3 -std=c++17 -Wall -Wshadow -Wextra -ffast-math -g -fopenmp -march=skylake-avx512 -mprefer-vector-width=256 -DMGONGPU_FPTYPE_DOUBLE -DMGONGPU_FPTYPE2_DOUBLE -DMGONGPU_HAS_NO_CURAND

%.o : %.cc
	g++ -I. -I../../src ${FLAGS} -fPIC -c $< -o $@

check.exe : check_sa.o CommonRandomNumberKernel.o RamboSamplingKernels.o CurandRandomNumberKernel.o ../../src/Parameters_sm.o ../../src/read_slha.o CPPProcess.o MatrixElementKernels.o BridgeKernels.o CrossSectionKernels.o fbridge.o 
	g++ -o check.exe -fopenmp -ldl -pthread $^
