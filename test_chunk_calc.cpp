#include <cstdio>

int main() {
    auto calc_params = [](unsigned int input_size) {
        unsigned int chunk_size = (input_size < 1024 * 1024) ? 64 * 1024 :
                                   (input_size < 10 * 1024 * 1024) ? 256 * 1024 :
                                   1024 * 1024;
        unsigned int num_chunks = (input_size + chunk_size - 1) / chunk_size;
        int max_passes = 5 + (num_chunks / 50);
        
        printf("Input: %8.2f MB | Chunk: %6u KB | Chunks: %5u | Passes: %3d | Total kernel launches: %5d\n",
               input_size / (1024.0 * 1024.0),
               chunk_size / 1024,
               num_chunks,
               max_passes,
               num_chunks * max_passes);
    };
    
    printf("Chunk Size and Pass Count Analysis:\n");
    printf("===================================\n");
    calc_params(512 * 1024);        // 512KB
    calc_params(1024 * 1024);       // 1MB
    calc_params(10 * 1024 * 1024);  // 10MB  
    calc_params(100 * 1024 * 1024); // 100MB
    
    return 0;
}
