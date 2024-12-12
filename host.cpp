// host.cpp

// #include <iostream>
// #include <cstdlib>
// #include <cstdint>
// #include <ap_fixed.h>

#include "resnet18.h"
#include "quantize.h"
#include "block_function.h"


int main() {
    // Generate random 8-bit input
    static data_t input_block[INPUT_TENSOR_SIZE];
    for (int i = 0; i < INPUT_TENSOR_SIZE; i++) {
        // input_block[i] = float(rand()) % 128.0f; // random 8-bit signed
        float random_val = ((rand() % 160) / 10.0f) - 8.0f; // -8.0 to +7.9
        input_block[i] = data_t(random_val);
    }

    // Output buffer
    static data_t output_block[OUTPUT_TENSOR_SIZE];

    // Run the block
    run_resnet_block(
        input_block, 
        output_block,
        IN_H, IN_W, IN_C, OUT_C
    );

    // Print some portion of the output for verification
    // for (int i = 0; i < 10; i++) {
    //     std::cout << (int)output_block[i] << " ";
    // }
    // std::cout << std::endl;
    printf("//I am here/// \n");

    std::cout << "Block computation complete." << std::endl;
    return 0;
}
