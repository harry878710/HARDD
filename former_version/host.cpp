// host.cpp

// #include <iostream>
// #include <cstdlib>
// #include <cstdint>
// #include <ap_fixed.h>
#include "resnet18.h"
#include "quantize.h"
#include "block_function.h"
using namespace std;
// extern float   block_conv1_bn_mean[];
// extern data_t input_block[];

// Define the arrays as per the header
data_t input_block[INPUT_TENSOR_SIZE]; // 14x14x256
data_t output_block[OUTPUT_TENSOR_SIZE]; // 7x7x512
data_t block_conv1_weights[BLOCK_CONV1_SIZE]; // 512*256*3*3
// acc_t block_conv1_bias[BLOCK_CONV1_OUT_C];
float block_conv1_bn_mean[OUT_C]; // 512
float block_conv1_bn_deno[OUT_C];
float block_conv1_bn_gamma[OUT_C];
float block_conv1_bn_beta[OUT_C];

data_t block_conv2_weights[BLOCK_CONV2_SIZE]; // 512*512*3*3
// acc_t block_conv2_bias[BLOCK_CONV2_OUT_C];
float block_conv2_bn_mean[OUT_C];
float block_conv2_bn_deno[OUT_C];
float block_conv2_bn_gamma[OUT_C];
float block_conv2_bn_beta[OUT_C];

data_t skip_conv_weights[SKIP_CONV_SIZE]; // 512*256*3*3
// acc_t skip_conv_bias[SKIP_CONV_OUT_C];
float skip_bn_mean[OUT_C];
float skip_bn_deno[OUT_C];
float skip_bn_gamma[OUT_C];
float skip_bn_beta[OUT_C];

static bool arrays_initialized = false;
template <typename T>
bool read_data_from_file(const char* filename, T* arr, int size);
bool initialize_arrays();

int main() {
    // initialize arrays for inputs
    if (!initialize_arrays()){
        return -1; // Initialization failed
    }

    // verify the input feature map
    for (int i = 0; i < 100; i++) {
        printf("Input featuremap[%d]: %f\n", i, (float)input_block[i]);
    }

    // Create an hls::streams
    // hls::stream<float> data_stream;

    // Load data into the streams
    // if (!load_array_to_stream(data_stream, data_array, DATA_SIZE)) {
    //     std::cerr << "Error: Failed to load data into stream." << std::endl;
    //     return -1;
    // }
    // std::cout << "Data successfully loaded into hls::stream." << std::endl;

    // Generate random 8-bit input
    // static data_t input_block[INPUT_TENSOR_SIZE];
    // for (int i = 0; i < INPUT_TENSOR_SIZE; i++) {
    //     // input_block[i] = float(rand()) % 128.0f; // random 8-bit signed
    //     float random_val = ((rand() % 160) / 10.0f) - 8.0f; // -8.0 to +7.9
    //     input_block[i] = data_t(random_val);
    // }

    // Output buffer
    // static data_t output_block[OUTPUT_TENSOR_SIZE];

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
        printf("Output tensor [%d]: %f\n", i, (float)output_block[i]);
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

    // Read the file and print its contents
    // while (getline(file, line)){
    //     cout << line << endl;
    // }
    // file.close();

    return 0;
}

// Function to read data from a file into an array
template <typename T>
bool read_data_from_file(const char* filename, T* arr, int size) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    for (int i = 0; i < size; ++i) {
        if (!(infile >> arr[i])) {
            std::cerr << "Error: Not enough data in file." << std::endl;
            return false;
        }
    }

    infile.close();
    return true;
}

// initializes all arrays by reading from text files
bool initialize_arrays() {
    if (arrays_initialized) {
        return true; // Already initialized
    }

    bool success = true;

    success &= read_data_from_file("resnet18_input_data.txt", input_block, INPUT_TENSOR_SIZE);
    success &= read_data_from_file("block_conv1_weights.txt", block_conv1_weights, BLOCK_CONV1_SIZE);
    // success &= read_data_from_file("block_conv1_bias.txt", block_conv1_bias, BLOCK_CONV1_OUT_C);
    success &= read_data_from_file("block_conv1_bn_mean.txt", block_conv1_bn_mean, OUT_C);
    success &= read_data_from_file("block_conv1_bn_deno.txt", block_conv1_bn_deno, OUT_C);
    success &= read_data_from_file("block_conv1_bn_gamma.txt", block_conv1_bn_gamma, OUT_C);
    success &= read_data_from_file("block_conv1_bn_beta.txt", block_conv1_bn_beta, OUT_C);

    success &= read_data_from_file("block_conv2_weights.txt", block_conv2_weights, BLOCK_CONV2_SIZE);
    // success &= read_data_from_file("block_conv2_bias.txt", block_conv2_bias, BLOCK_CONV2_OUT_C);
    success &= read_data_from_file("block_conv2_bn_mean.txt", block_conv2_bn_mean, OUT_C);
    success &= read_data_from_file("block_conv2_bn_deno.txt", block_conv2_bn_deno, OUT_C);
    success &= read_data_from_file("block_conv2_bn_gamma.txt", block_conv2_bn_gamma, OUT_C);
    success &= read_data_from_file("block_conv2_bn_beta.txt", block_conv2_bn_beta, OUT_C);

    success &= read_data_from_file("skip_conv_weights.txt", skip_conv_weights, SKIP_CONV_SIZE);
    // success &= read_data_from_file("skip_conv_bias.txt", skip_conv_bias, SKIP_CONV_OUT_C);
    success &= read_data_from_file("skip_bn_mean.txt", skip_bn_mean, OUT_C);
    success &= read_data_from_file("skip_bn_deno.txt", skip_bn_deno, OUT_C);
    success &= read_data_from_file("skip_bn_gamma.txt", skip_bn_gamma, OUT_C);
    success &= read_data_from_file("skip_bn_beta.txt", skip_bn_beta, OUT_C);

    if (!success) {
        std::cerr << "Error: Failed to load one or more weight/BN files." << std::endl;
        return false;
    }

    arrays_initialized = true;
    return true;
}
