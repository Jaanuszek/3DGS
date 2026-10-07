#include <stdio.h>
#include <iostream>

__global__ void testKernel()
{
    int id = threadIdx.x;

    printf("siema eniu: %d\n", id);
}

int main()
{
    testKernel<<<1, 256>>>();

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "WTFFF " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
    return 0;
}
