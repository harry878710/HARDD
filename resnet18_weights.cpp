// resnet18_weights.cpp
#include "resnet18_weights.h"
#include <string>
using namespace std;

static bool arrays_initialized = false;

template<typename T>
bool load_array_from_file(const string &filename, T* arr, int size) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error: Unable to open " << filename << endl;
        return false;
    }
    for (int i = 0; i < size; i++) {
        if (!(infile >> arr[i])) {
            cerr << "Error: Not enough data in " << filename << endl;
            return false;
        }
    }
    infile.close();
    return true;
}

// Define the arrays as per the header
// data_t block_conv1_weights[BLOCK_CONV1_SIZE];
// acc_t block_conv1_bias[BLOCK_CONV1_OUT_C];
float block_conv1_bn_mean[OUT_C];
// float block_conv1_bn_deno[OUT_C];
// float block_conv1_bn_gamma[OUT_C];
// float block_conv1_bn_beta[OUT_C];

// data_t block_conv2_weights[BLOCK_CONV2_SIZE];
// acc_t block_conv2_bias[BLOCK_CONV2_OUT_C];
// float block_conv2_bn_mean[OUT_C];
// float block_conv2_bn_deno[OUT_C];
// float block_conv2_bn_gamma[OUT_C];
// float block_conv2_bn_beta[OUT_C];

// data_t skip_conv_weights[SKIP_CONV_SIZE];
// acc_t skip_conv_bias[SKIP_CONV_OUT_C];
// float skip_bn_mean[OUT_C];
// float skip_bn_deno[OUT_C];
// float skip_bn_gamma[OUT_C];
// float skip_bn_beta[OUT_C];

// This function initializes all arrays by reading from text files.
// Call this function before running your FPGA block or main inference function.
bool initialize_arrays() {
    if (arrays_initialized) {
        return true; // Already initialized
    }

    bool success = true;

    // success &= load_array_from_file("block_conv1_weights.txt", block_conv1_weights, BLOCK_CONV1_SIZE);
    // success &= load_array_from_file("block_conv1_bias.txt", block_conv1_bias, BLOCK_CONV1_OUT_C);
    success &= load_array_from_file("./input_data/block_conv1_bn_mean.txt", block_conv1_bn_mean, OUT_C);
    // success &= load_array_from_file("block_conv1_bn_deno.txt", block_conv1_bn_deno, OUT_C);
    // success &= load_array_from_file("block_conv1_bn_gamma.txt", block_conv1_bn_gamma, OUT_C);
    // success &= load_array_from_file("block_conv1_bn_beta.txt", block_conv1_bn_beta, OUT_C);

    // success &= load_array_from_file("block_conv2_weights.txt", block_conv2_weights, BLOCK_CONV2_SIZE);
    // success &= load_array_from_file("block_conv2_bias.txt", block_conv2_bias, BLOCK_CONV2_OUT_C);
    // success &= load_array_from_file("block_conv2_bn_mean.txt", block_conv2_bn_mean, OUT_C);
    // success &= load_array_from_file("block_conv2_bn_deno.txt", block_conv2_bn_deno, OUT_C);
    // success &= load_array_from_file("block_conv2_bn_gamma.txt", block_conv2_bn_gamma, OUT_C);
    // success &= load_array_from_file("block_conv2_bn_beta.txt", block_conv2_bn_beta, OUT_C);

    // success &= load_array_from_file("skip_conv_weights.txt", skip_conv_weights, SKIP_CONV_SIZE);
    // success &= load_array_from_file("skip_conv_bias.txt", skip_conv_bias, SKIP_CONV_OUT_C);
    // success &= load_array_from_file("skip_bn_mean.txt", skip_bn_mean, OUT_C);
    // success &= load_array_from_file("skip_bn_deno.txt", skip_bn_deno, OUT_C);
    // success &= load_array_from_file("skip_bn_gamma.txt", skip_bn_gamma, OUT_C);
    // success &= load_array_from_file("skip_bn_beta.txt", skip_bn_beta, OUT_C);

    if (!success) {
        cerr << "Error: Failed to load one or more weight/BN files." << endl;
        return false;
    }

    arrays_initialized = true;
    return true;
}
