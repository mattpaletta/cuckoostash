#include "cuckoo.hpp"

#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <stdio.h>

#include "get_value.cu"

//template<std::size_t N = 1024>
Cuckoo::Cuckoo() {
	for (Entry i = 0; i < CUCKOO_SIZE; ++i) {
		this->ccuckoo[i] = SLOT_EMPTY;
	}
	for (Entry i = 0; i < STASH_SIZE; ++i) {
		this->cstash[i] = SLOT_EMPTY;
	}

	std::cout << "Malloc arrays" << std::endl;
	auto r1 = cudaMalloc((void **) &gcuckoo, CUCKOO_SIZE * sizeof(Entry));	
	auto r2 = cudaMalloc((void **) &gstash, STASH_SIZE * sizeof(Entry));
	std::cout << "Allocated" << r1 << r2 << std::endl;
	std::cout << "Copying to cuckoo device" << std::endl;
	cudaMemcpy(gcuckoo, this->ccuckoo, CUCKOO_SIZE * sizeof(Entry), cudaMemcpyHostToDevice);
	std::cout << "Copying to stash device" << std::endl;
	cudaMemcpy(gstash, this->cstash, CUCKOO_SIZE * sizeof(Entry), cudaMemcpyHostToDevice);
}

Cuckoo::~Cuckoo() {
	cudaFree(gcuckoo);
	cudaFree(gstash);
}

// Run test on Cuda
int Cuckoo::set() {
	std::cout << "New Declaration!" << std::endl;

	//setValue<<<dim3(32, 1, 1), dim3(32, 1, 1)>>>(gpuFloats, 2, 123.0f);

	//float hostFloats[4];
	//cudaMemcpy(hostFloats, gpuFloats, 4 * sizeof(float), cudaMemcpyDeviceToHost);
	//std::cout << "hostFloats[2] " << hostFloats[2] << std::endl;

	//setValue<<<dim3(32, 1, 1), dim3(32, 1, 1)>>>(gpuFloats, 2, 222.0f);
	//cudaMemcpy(hostFloats, gpuFloats, 4 * sizeof(float), cudaMemcpyDeviceToHost);
	//std::cout << "hostFloats[2] " << hostFloats[2] << std::endl;

	//hostFloats[2] = 444.0f;
	//cudaMemcpy(gpuFloats, hostFloats, 4 * sizeof(float), cudaMemcpyHostToDevice);
	//hostFloats[2] = 555.0f;
	//cudaMemcpy(hostFloats, gpuFloats, 4 * sizeof(float), cudaMemcpyDeviceToHost);
	//std::cout << "hostFloats[2] " << hostFloats[2] << std::endl;

	return 0;
}
