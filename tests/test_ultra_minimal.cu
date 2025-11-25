#include <iostream>
#include <cuda_runtime.h>

int main() {
    std::cout << "Step 1: Starting" << std::endl;
    
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "Step 2: Device count = " << device_count << std::endl;
    
    std::cout << "Step 3: Exiting normally" << std::endl;
    return 0;
}
