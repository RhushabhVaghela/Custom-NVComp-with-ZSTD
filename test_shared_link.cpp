#include "cuda_zstd_manager.h"
#include <iostream>

int main() {
    auto manager = cuda_zstd::create_manager();
    if (manager) {
        std::cout << "Manager created successfully" << std::endl;
        // unique_ptr handles destruction
    } else {
        std::cerr << "Failed to create manager" << std::endl;
        return 1;
    }
    return 0;
}
