#include "utils.cuh"
#include <stdio.h>

void gpuAssert(cudaError_t code)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",
                cudaGetErrorString(code), __PRETTY_FUNCTION__, __LINE__,
                __FILE__);                                              
        exit(-1);    
    }
}