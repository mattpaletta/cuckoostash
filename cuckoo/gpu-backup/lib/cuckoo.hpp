#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <stdio.h>

#include "get_value.cu"

// Run test on Cuda
int Cuckoo::set() {
    std::cout << "New Declaration!" << std::endl;

    int N = 1024;

    float *gpuFloats;
    cudaMalloc((void**)(&gpuFloats), N * sizeof(float));

    setValue<<<dim3(32, 1, 1), dim3(32, 1, 1)>>>(gpuFloats, 2, 123.0f);

    float hostFloats[4];
    cudaMemcpy(hostFloats, gpuFloats, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "hostFloats[2] " << hostFloats[2] << std::endl;

    setValue<<<dim3(32, 1, 1), dim3(32, 1, 1)>>>(gpuFloats, 2, 222.0f);
    cudaMemcpy(hostFloats, gpuFloats, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "hostFloats[2] " << hostFloats[2] << std::endl;

    hostFloats[2] = 444.0f;
    cudaMemcpy(gpuFloats, hostFloats, 4 * sizeof(float), cudaMemcpyHostToDevice);
    hostFloats[2] = 555.0f;
    cudaMemcpy(hostFloats, gpuFloats, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "hostFloats[2] " << hostFloats[2] << std::endl;

    cudaFree(gpuFloats);

    std::cout << "Hello World!" << std::endl;

    return 0;
}

int main() {
    Cuckoo c = Cuckoo();
    c.set();
    return 0;
}