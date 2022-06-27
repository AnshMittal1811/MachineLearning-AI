#include <cuda.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans)); }

void gpuAssert(cudaError_t code);