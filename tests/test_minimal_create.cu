#include "cuda_zstd_manager.h"
#include <iostream>

int main() {
    std::cout << "Step 1: Creating manager..." << std::endl;
    
    {
        auto manager = cuda_zstd::create_manager(5);
        
        std::cout << "Step 2: Manager created successfully!" << std::endl;
        
        // Manager will be destroyed when going out of scope
        std::cout << "Step 3: About to destroy manager..." << std::endl;
    }
    
    std::cout << "Step 4: Manager destroyed, exiting..." << std::endl;
    
    return 0;
}
