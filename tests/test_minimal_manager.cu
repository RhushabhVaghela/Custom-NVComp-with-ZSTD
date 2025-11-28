#include "cuda_zstd_manager.h"
#include <stdio.h>

int main() {
    printf("Creating manager...\n");
    fflush(stdout);
    
    auto manager = cuda_zstd::create_manager(5);
    
    printf("Manager created successfully!\n");
    fflush(stdout);
    
    return 0;
}
