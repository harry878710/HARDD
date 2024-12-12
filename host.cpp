// host.cpp

// #include <iostream>
// #include <cstdlib>
// #include <cstdint>
// #include <ap_fixed.h>

#include "resnet18.h"
#include "quantize.h"
#include "block_function.h"
using namespace std;
extern float   block_conv1_bn_mean[];
int main() {
    // Set input file path
    // ifstream file("output_tensor.txt");
    // string line;
    if (!initialize_arrays()){
        return -1; // Initialization failed
    }

    for (int i = 0; i < OUT_C; i++) {
        printf("Mean[%d]: %f\n", i, block_conv1_bn_mean[i]);
    }

    // Generate random 8-bit input
    static data_t input_block[INPUT_TENSOR_SIZE];
    for (int i = 0; i < INPUT_TENSOR_SIZE; i++) {
        // input_block[i] = float(rand()) % 128.0f; // random 8-bit signed
        float random_val = ((rand() % 160) / 10.0f) - 8.0f; // -8.0 to +7.9
        input_block[i] = data_t(random_val);
    }

    // Output buffer
    static data_t output_block[OUTPUT_TENSOR_SIZE];
/*
    // Run the block
    run_resnet_block(
        input_block, 
        output_block,
        IN_H, IN_W, IN_C, OUT_C
    );
    printf("Block computation complete.\n");
    // printf("//I am here/// \n");

    // Print some portion of the output for verification
    for (int i = 0; i < 10; i++) {
        printf("Output tensor [%d]: %d\n", i, (int)output_block[i]);
    }

    // Write out output tensor
    ofstream outFile("output_tensor.txt"); 
    if (!outFile){
        cerr << "Error: Could not open the file for writing." << endl;
        return 1;
    }
    for (int i = 0; i < OUTPUT_TENSOR_SIZE; i++) {
        outFile << output_block[i] << " ";
    }
    outFile << endl;
    outFile.close();
    cout << "Output tensor written to output_tensor.txt successfully." << endl;
*/
    // Read the file and print its contents
    // while (getline(file, line)){
    //     cout << line << endl;
    // }
    // file.close();

    return 0;
}
