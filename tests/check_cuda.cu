#include <cuda_runtime.h>
#include <iostream>

int main() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << " (" << error << ")" << std::endl;
        return 1;
    }
    std::cout << "CUDA devices: " << device_count << std::endl;
    return 0;
}
