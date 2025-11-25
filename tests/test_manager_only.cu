#include "cuda_zstd_manager.h"
#include <iostream>

int main() {
    std::cout << "Creating manager..." << std::endl;
    auto manager = cuda_zstd::create_manager(5);
    std::cout << "Manager created!" << std::endl;
    return 0;
}
