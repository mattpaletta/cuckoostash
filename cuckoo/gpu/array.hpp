#include <iostream>
#include <cuda_runtime.h>
template<class T = int>
class GPUArray {
public:
	GPUArray(T* data, const std::size_t N) {
		this->cpu_data = data;
		this->gpu_data = new T[N];
		this->N = N;
		this->gpu_is_updated = false;
		this->cpu_is_updated = true;
		cudaMalloc(&gpu_data, N * sizeof(T)); 
	}

	~GPUArray() {
		cudaFree(this->gpu_data);
		// delete[](this->gpu_data);
		// delete[](this->cpu_data);
	}

	T* get_cpu() {
		gpu_is_updated = false;
		return this->cpu_data;
	}

	T* to_gpu() {
		if (!this->gpu_is_updated) {
			std::cout << "Copying to GPU" << std::endl;
			cudaMemcpy(this->gpu_data, this->cpu_data, this->N * sizeof(T), cudaMemcpyHostToDevice);
			this->gpu_is_updated = true;
			this->cpu_is_updated = false;
		}
		return gpu_data;
	}

	T* get_gpu() {
		std::cout << "Copying from GPU" << std::endl;
		if (!this->cpu_is_updated) {
			cudaMemcpy(cpu_data, gpu_data, this->N * sizeof(T), cudaMemcpyDeviceToHost);
			cpu_is_updated = true;
		}
		return cpu_data;
	}

private:
	bool gpu_is_updated = false;
	bool cpu_is_updated = true;
	std::size_t N;
	T* cpu_data;
	T* gpu_data;
};
