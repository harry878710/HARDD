// host.cpp

// #include <iostream>
// #include <cstdlib>
// #include <cstdint>
// #include <ap_fixed.h>
#include "resnet18.h"
#include "quantize.h"
#include "ResNet18_top.h"
using namespace std;

// Define the arrays as per the header
data_t input_block[INPUT_TENSOR_SIZE];   // 224 224 3
data_t output_block[OUTPUT_TENSOR_SIZE]; // 1 1 1000

float f_input_block[INPUT_TENSOR_SIZE];
float f_output_block[OUTPUT_TENSOR_SIZE];

//-------------------------------------------------
// Dimension L0
// L0 Block 0
// acc_t block_conv1_bias[BLOCK_CONV1_OUT_C];
float f_L0_block_conv1_weights[L0_BLOCK_CONV1_SIZE];
data_t L0_block_conv1_weights[L0_BLOCK_CONV1_SIZE];
float L0_block_conv1_bn_mean[L0_OUT_C];
float L0_block_conv1_bn_deno[L0_OUT_C];
float L0_block_conv1_bn_gamma[L0_OUT_C];
float L0_block_conv1_bn_beta[L0_OUT_C];

//---------------------------------------------------------------
// Dimension L1
// L1 Block 0
// acc_t block_conv1_bias[BLOCK_CONV1_OUT_C];
float f_L1_block0_conv1_weights[L1_BLOCK0_CONV1_SIZE];
data_t L1_block0_conv1_weights[L1_BLOCK0_CONV1_SIZE];
float L1_block0_conv1_bn_mean[L1_OUT_C];
float L1_block0_conv1_bn_deno[L1_OUT_C];
float L1_block0_conv1_bn_gamma[L1_OUT_C];
float L1_block0_conv1_bn_beta[L1_OUT_C];

data_t L1_block0_conv2_weights[L1_BLOCK0_CONV2_SIZE];
float f_L1__block0_conv2_weights[L1_BLOCK0_CONV2_SIZE];
// acc_t block_conv2_bias[BLOCK_CONV2_OUT_C];
float L1_block0_conv2_bn_mean[L1_OUT_C];
float L1_block0_conv2_bn_deno[L1_OUT_C];
float L1_block0_conv2_bn_gamma[L1_OUT_C];
float L1_block0_conv2_bn_beta[L1_OUT_C];

// L1 Block 1
float f_L1_block1_conv1_weights[L1_BLOCK1_CONV1_SIZE];
data_t L1_block1_conv1_weights[L1_BLOCK1_CONV1_SIZE];
float L1_block1_conv1_bn_mean[L1_OUT_C];
float L1_block1_conv1_bn_deno[L1_OUT_C];
float L1_block1_conv1_bn_gamma[L1_OUT_C];
float L1_block1_conv1_bn_beta[L1_OUT_C];

data_t L1_block1_conv2_weights[L1_BLOCK1_CONV2_SIZE];
float f_L1_block1_conv2_weights[L1_BLOCK1_CONV2_SIZE];
// acc_t block_conv2_bias[BLOCK_CONV2_OUT_C];
float L1_block1_conv2_bn_mean[L1_OUT_C];
float L1_block1_conv2_bn_deno[L1_OUT_C];
float L1_block1_conv2_bn_gamma[L1_OUT_C];
float L1_block1_conv2_bn_beta[L1_OUT_C];

//---------------------------------------------------------------
// Dimension L2
// L2 Block 0
// acc_t block_conv1_bias[BLOCK_CONV1_OUT_C];
float f_L2_block0_conv1_weights[L2_BLOCK0_CONV1_SIZE];
data_t L2_block0_conv1_weights[L2_BLOCK0_CONV1_SIZE];
float L2_block0_conv1_bn_mean[L2_OUT_C];
float L2_block0_conv1_bn_deno[L2_OUT_C];
float L2_block0_conv1_bn_gamma[L2_OUT_C];
float L2_block0_conv1_bn_beta[L2_OUT_C];

data_t L2_block0_conv2_weights[L2_BLOCK0_CONV2_SIZE];
float f_L2__block0_conv2_weights[L2_BLOCK0_CONV2_SIZE];
// acc_t block_conv2_bias[BLOCK_CONV2_OUT_C];
float L2_block0_conv2_bn_mean[L2_OUT_C];
float L2_block0_conv2_bn_deno[L2_OUT_C];
float L2_block0_conv2_bn_gamma[L2_OUT_C];
float L2_block0_conv2_bn_beta[L2_OUT_C];

data_t L2_skip_conv_weights[L2_SKIP_CONV_SIZE];
float f_L2_skip_conv_weights[L2_SKIP_CONV_SIZE];
// acc_t skip_conv_bias[SKIP_CONV_OUT_C];
float L2_skip_bn_mean[L2_OUT_C];
float L2_skip_bn_deno[L2_OUT_C];
float L2_skip_bn_gamma[L2_OUT_C];
float L2_skip_bn_beta[L2_OUT_C];

// L2 Block 1
float f_L2_block1_conv1_weights[L2_BLOCK1_CONV1_SIZE];
data_t L2_block1_conv1_weights[L2_BLOCK1_CONV1_SIZE];
float L2_block1_conv1_bn_mean[L2_OUT_C];
float L2_block1_conv1_bn_deno[L2_OUT_C];
float L2_block1_conv1_bn_gamma[L2_OUT_C];
float L2_block1_conv1_bn_beta[L2_OUT_C];

data_t L2_block1_conv2_weights[L2_BLOCK1_CONV2_SIZE];
float f_L2_block1_conv2_weights[L2_BLOCK1_CONV2_SIZE];
// acc_t block_conv2_bias[BLOCK_CONV2_OUT_C];
float L2_block1_conv2_bn_mean[L2_OUT_C];
float L2_block1_conv2_bn_deno[L2_OUT_C];
float L2_block1_conv2_bn_gamma[L2_OUT_C];
float L2_block1_conv2_bn_beta[L2_OUT_C];

//---------------------------------------------------------------
// Dimension L3
// L3 Block 0
// acc_t block_conv1_bias[BLOCK_CONV1_OUT_C];
float f_L3_block0_conv1_weights[L3_BLOCK0_CONV1_SIZE];
data_t L3_block0_conv1_weights[L3_BLOCK0_CONV1_SIZE];
float L3_block0_conv1_bn_mean[L3_OUT_C];
float L3_block0_conv1_bn_deno[L3_OUT_C];
float L3_block0_conv1_bn_gamma[L3_OUT_C];
float L3_block0_conv1_bn_beta[L3_OUT_C];

data_t L3_block0_conv2_weights[L3_BLOCK0_CONV2_SIZE];
float f_L3__block0_conv2_weights[L3_BLOCK0_CONV2_SIZE];
// acc_t block_conv2_bias[BLOCK_CONV2_OUT_C];
float L3_block0_conv2_bn_mean[L3_OUT_C];
float L3_block0_conv2_bn_deno[L3_OUT_C];
float L3_block0_conv2_bn_gamma[L3_OUT_C];
float L3_block0_conv2_bn_beta[L3_OUT_C];

data_t L3_skip_conv_weights[L3_SKIP_CONV_SIZE];
float f_L3_skip_conv_weights[L3_SKIP_CONV_SIZE];
// acc_t skip_conv_bias[SKIP_CONV_OUT_C];
float L3_skip_bn_mean[L3_OUT_C];
float L3_skip_bn_deno[L3_OUT_C];
float L3_skip_bn_gamma[L3_OUT_C];
float L3_skip_bn_beta[L3_OUT_C];

// L3 Block 1
float f_L3_block1_conv1_weights[L3_BLOCK1_CONV1_SIZE];
data_t L3_block1_conv1_weights[L3_BLOCK1_CONV1_SIZE];
float L3_block1_conv1_bn_mean[L3_OUT_C];
float L3_block1_conv1_bn_deno[L3_OUT_C];
float L3_block1_conv1_bn_gamma[L3_OUT_C];
float L3_block1_conv1_bn_beta[L3_OUT_C];

data_t L3_block1_conv2_weights[L3_BLOCK1_CONV2_SIZE];
float f_L3_block1_conv2_weights[L3_BLOCK1_CONV2_SIZE];
// acc_t block_conv2_bias[BLOCK_CONV2_OUT_C];
float L3_block1_conv2_bn_mean[L3_OUT_C];
float L3_block1_conv2_bn_deno[L3_OUT_C];
float L3_block1_conv2_bn_gamma[L3_OUT_C];
float L3_block1_conv2_bn_beta[L3_OUT_C];

//---------------------------------------------------------------

// Dimension L4
// L4 Block 0
// acc_t block_conv1_bias[BLOCK_CONV1_OUT_C];
float f_L4_block0_conv1_weights[L4_BLOCK0_CONV1_SIZE];
data_t L4_block0_conv1_weights[L4_BLOCK0_CONV1_SIZE];
float L4_block0_conv1_bn_mean[L4_OUT_C];
float L4_block0_conv1_bn_deno[L4_OUT_C];
float L4_block0_conv1_bn_gamma[L4_OUT_C];
float L4_block0_conv1_bn_beta[L4_OUT_C];

data_t L4_block0_conv2_weights[L4_BLOCK0_CONV2_SIZE];
float f_L4__block0_conv2_weights[L4_BLOCK0_CONV2_SIZE];
// acc_t block_conv2_bias[BLOCK_CONV2_OUT_C];
float L4_block0_conv2_bn_mean[L4_OUT_C];
float L4_block0_conv2_bn_deno[L4_OUT_C];
float L4_block0_conv2_bn_gamma[L4_OUT_C];
float L4_block0_conv2_bn_beta[L4_OUT_C];

data_t L4_skip_conv_weights[L4_SKIP_CONV_SIZE];
float f_L4_skip_conv_weights[L4_SKIP_CONV_SIZE];
// acc_t skip_conv_bias[SKIP_CONV_OUT_C];
float L4_skip_bn_mean[L4_OUT_C];
float L4_skip_bn_deno[L4_OUT_C];
float L4_skip_bn_gamma[L4_OUT_C];
float L4_skip_bn_beta[L4_OUT_C];

// L4 Block 1
float f_L4_block1_conv1_weights[L4_BLOCK1_CONV1_SIZE];
data_t L4_block1_conv1_weights[L4_BLOCK1_CONV1_SIZE];
float L4_block1_conv1_bn_mean[L4_OUT_C];
float L4_block1_conv1_bn_deno[L4_OUT_C];
float L4_block1_conv1_bn_gamma[L4_OUT_C];
float L4_block1_conv1_bn_beta[L4_OUT_C];

data_t L4_block1_conv2_weights[L4_BLOCK1_CONV2_SIZE];
float f_L4_block1_conv2_weights[L4_BLOCK1_CONV2_SIZE];
// acc_t block_conv2_bias[BLOCK_CONV2_OUT_C];
float L4_block1_conv2_bn_mean[L4_OUT_C];
float L4_block1_conv2_bn_deno[L4_OUT_C];
float L4_block1_conv2_bn_gamma[L4_OUT_C];
float L4_block1_conv2_bn_beta[L4_OUT_C];

// Fully connected layer
data_t fc_weights[FC_WEIGHTS_SIZE];
float f_fc_weights[FC_WEIGHTS_SIZE];
data_t fc_bias[FC_OUT_C];
float f_fc_bias[FC_OUT_C];

//---------------------------------------------------------------
static bool arrays_initialized = false;
template <typename T>
bool read_data_from_file(const char *filename, T *arr, int size);
bool initialize_arrays();

int main()
{
    // ============================================================================
    // initialize arrays for inputs
    // ============================================================================
    if (!initialize_arrays())
    {
        return -1; // Initialization failed
    }
    // ============================================================================
    // Quantize input feature map & weights to data type (e.g. ap_fixed<8,4>)
    // ============================================================================
    for (int i = 0; i < INPUT_TENSOR_SIZE; i++)
        input_block[i] = quantize_float_to_data(f_input_block[i]);

    // L0
    for (int i = 0; i < L0_BLOCK_CONV1_SIZE; i++)
        L0_block_conv1_weights[i] = quantize_float_to_data(f_L0_block_conv1_weights[i]);
    // --------------------------------
    // L1 Block 0
    for (int i = 0; i < L1_BLOCK0_CONV1_SIZE; i++)
        L1_block0_conv1_weights[i] = quantize_float_to_data(f_L1_block0_conv1_weights[i]);
    for (int i = 0; i < L1_BLOCK0_CONV2_SIZE; i++)
        L1_block0_conv2_weights[i] = quantize_float_to_data(f_L1__block0_conv2_weights[i]);

    // L1 Block 1
    for (int i = 0; i < L1_BLOCK1_CONV1_SIZE; i++)
        L1_block1_conv1_weights[i] = quantize_float_to_data(f_L1_block1_conv1_weights[i]);
    for (int i = 0; i < L1_BLOCK1_CONV2_SIZE; i++)
        L1_block1_conv2_weights[i] = quantize_float_to_data(f_L1_block1_conv2_weights[i]);
    //---------------------------------------
    // L2 Block 0
    for (int i = 0; i < L2_BLOCK0_CONV1_SIZE; i++)
        L2_block0_conv1_weights[i] = quantize_float_to_data(f_L2_block0_conv1_weights[i]);
    for (int i = 0; i < L2_BLOCK0_CONV2_SIZE; i++)
        L2_block0_conv2_weights[i] = quantize_float_to_data(f_L2__block0_conv2_weights[i]);

    // L2 Skip
    for (int i = 0; i < L2_SKIP_CONV_SIZE; i++)
        L2_skip_conv_weights[i] = quantize_float_to_data(f_L2_skip_conv_weights[i]);

    // L2 Block 1
    for (int i = 0; i < L2_BLOCK1_CONV1_SIZE; i++)
        L2_block1_conv1_weights[i] = quantize_float_to_data(f_L2_block1_conv1_weights[i]);
    for (int i = 0; i < L2_BLOCK1_CONV2_SIZE; i++)
        L2_block1_conv2_weights[i] = quantize_float_to_data(f_L2_block1_conv2_weights[i]);
    //---------------------------------------
    // L3 Block 0
    for (int i = 0; i < L3_BLOCK0_CONV1_SIZE; i++)
        L3_block0_conv1_weights[i] = quantize_float_to_data(f_L3_block0_conv1_weights[i]);
    for (int i = 0; i < L3_BLOCK0_CONV2_SIZE; i++)
        L3_block0_conv2_weights[i] = quantize_float_to_data(f_L3__block0_conv2_weights[i]);

    // L3 Skip
    for (int i = 0; i < L3_SKIP_CONV_SIZE; i++)
        L3_skip_conv_weights[i] = quantize_float_to_data(f_L3_skip_conv_weights[i]);

    // L3 Block 1
    for (int i = 0; i < L3_BLOCK1_CONV1_SIZE; i++)
        L3_block1_conv1_weights[i] = quantize_float_to_data(f_L3_block1_conv1_weights[i]);
    for (int i = 0; i < L3_BLOCK1_CONV2_SIZE; i++)
        L3_block1_conv2_weights[i] = quantize_float_to_data(f_L3_block1_conv2_weights[i]);
    //---------------------------------------
    // L4 Block 0
    for (int i = 0; i < L4_BLOCK0_CONV1_SIZE; i++)
        L4_block0_conv1_weights[i] = quantize_float_to_data(f_L4_block0_conv1_weights[i]);
    for (int i = 0; i < L4_BLOCK0_CONV2_SIZE; i++)
        L4_block0_conv2_weights[i] = quantize_float_to_data(f_L4__block0_conv2_weights[i]);

    // L4 Skip
    for (int i = 0; i < L4_SKIP_CONV_SIZE; i++)
        L4_skip_conv_weights[i] = quantize_float_to_data(f_L4_skip_conv_weights[i]);

    // L4 Block 1
    for (int i = 0; i < L4_BLOCK1_CONV1_SIZE; i++)
        L4_block1_conv1_weights[i] = quantize_float_to_data(f_L4_block1_conv1_weights[i]);
    for (int i = 0; i < L4_BLOCK1_CONV2_SIZE; i++)
        L4_block1_conv2_weights[i] = quantize_float_to_data(f_L4_block1_conv2_weights[i]);

    // fc
    for (int i = 0; i < FC_WEIGHTS_SIZE; i++)
        fc_weights[i] = quantize_float_to_data(f_fc_weights[i]);
    for (int i = 0; i < FC_OUT_C; i++)
        fc_bias[i] = quantize_float_to_data(f_fc_bias[i]);

    // ============================================================================
    // Create an hls::streams
    // ============================================================================

    hls::stream<data_t> data_stream;
    hls::stream<data_t> output_stream;

    // Dimension L0
    // L0 Block 0
    // acc_t block_conv1_bias[BLOCK_CONV1_OUT_C];
    hls::stream<data_t> L0_block_conv1_weights_stream;
    hls::stream<float> L0_block_conv1_bn_mean_stream;
    hls::stream<float> L0_block_conv1_bn_deno_stream;
    hls::stream<float> L0_block_conv1_bn_gamma_stream;
    hls::stream<float> L0_block_conv1_bn_beta_stream;

    //---------------------------------------------------------------
    // Dimension L1
    // L1 Block 0
    hls::stream<data_t> L1_block0_conv1_weights_stream;
    hls::stream<float> L1_block0_conv1_bn_mean_stream;
    hls::stream<float> L1_block0_conv1_bn_deno_stream;
    hls::stream<float> L1_block0_conv1_bn_gamma_stream;
    hls::stream<float> L1_block0_conv1_bn_beta_stream;

    hls::stream<data_t> L1_block0_conv2_weights_stream;
    hls::stream<float> L1_block0_conv2_bn_mean_stream;
    hls::stream<float> L1_block0_conv2_bn_deno_stream;
    hls::stream<float> L1_block0_conv2_bn_gamma_stream;
    hls::stream<float> L1_block0_conv2_bn_beta_stream;

    // L1 Block 1
    hls::stream<data_t> L1_block1_conv1_weights_stream;
    hls::stream<float> L1_block1_conv1_bn_mean_stream;
    hls::stream<float> L1_block1_conv1_bn_deno_stream;
    hls::stream<float> L1_block1_conv1_bn_gamma_stream;
    hls::stream<float> L1_block1_conv1_bn_beta_stream;

    hls::stream<data_t> L1_block1_conv2_weights_stream;
    hls::stream<float> L1_block1_conv2_bn_mean_stream;
    hls::stream<float> L1_block1_conv2_bn_deno_stream;
    hls::stream<float> L1_block1_conv2_bn_gamma_stream;
    hls::stream<float> L1_block1_conv2_bn_beta_stream;
    //---------------------------------------------------------------
    // Dimension L2
    // L2 Block 0
    hls::stream<data_t> L2_block0_conv1_weights_stream;
    hls::stream<float> L2_block0_conv1_bn_mean_stream;
    hls::stream<float> L2_block0_conv1_bn_deno_stream;
    hls::stream<float> L2_block0_conv1_bn_gamma_stream;
    hls::stream<float> L2_block0_conv1_bn_beta_stream;

    hls::stream<data_t> L2_block0_conv2_weights_stream;
    hls::stream<float> L2_block0_conv2_bn_mean_stream;
    hls::stream<float> L2_block0_conv2_bn_deno_stream;
    hls::stream<float> L2_block0_conv2_bn_gamma_stream;
    hls::stream<float> L2_block0_conv2_bn_beta_stream;

    hls::stream<data_t> L2_skip_conv_weights_stream;
    hls::stream<float> L2_skip_bn_mean_stream;
    hls::stream<float> L2_skip_bn_deno_stream;
    hls::stream<float> L2_skip_bn_gamma_stream;
    hls::stream<float> L2_skip_bn_beta_stream;

    // L2 Block 1
    hls::stream<data_t> L2_block1_conv1_weights_stream;
    hls::stream<float> L2_block1_conv1_bn_mean_stream;
    hls::stream<float> L2_block1_conv1_bn_deno_stream;
    hls::stream<float> L2_block1_conv1_bn_gamma_stream;
    hls::stream<float> L2_block1_conv1_bn_beta_stream;

    hls::stream<data_t> L2_block1_conv2_weights_stream;
    hls::stream<float> L2_block1_conv2_bn_mean_stream;
    hls::stream<float> L2_block1_conv2_bn_deno_stream;
    hls::stream<float> L2_block1_conv2_bn_gamma_stream;
    hls::stream<float> L2_block1_conv2_bn_beta_stream;

    //---------------------------------------------------------------
    // Dimension L3
    // L3 Block 0
    hls::stream<data_t> L3_block0_conv1_weights_stream;
    hls::stream<float> L3_block0_conv1_bn_mean_stream;
    hls::stream<float> L3_block0_conv1_bn_deno_stream;
    hls::stream<float> L3_block0_conv1_bn_gamma_stream;
    hls::stream<float> L3_block0_conv1_bn_beta_stream;

    hls::stream<data_t> L3_block0_conv2_weights_stream;
    hls::stream<float> L3_block0_conv2_bn_mean_stream;
    hls::stream<float> L3_block0_conv2_bn_deno_stream;
    hls::stream<float> L3_block0_conv2_bn_gamma_stream;
    hls::stream<float> L3_block0_conv2_bn_beta_stream;

    hls::stream<data_t> L3_skip_conv_weights_stream;
    hls::stream<float> L3_skip_bn_mean_stream;
    hls::stream<float> L3_skip_bn_deno_stream;
    hls::stream<float> L3_skip_bn_gamma_stream;
    hls::stream<float> L3_skip_bn_beta_stream;

    // L3 Block 1
    hls::stream<data_t> L3_block1_conv1_weights_stream;
    hls::stream<float> L3_block1_conv1_bn_mean_stream;
    hls::stream<float> L3_block1_conv1_bn_deno_stream;
    hls::stream<float> L3_block1_conv1_bn_gamma_stream;
    hls::stream<float> L3_block1_conv1_bn_beta_stream;

    hls::stream<data_t> L3_block1_conv2_weights_stream;
    hls::stream<float> L3_block1_conv2_bn_mean_stream;
    hls::stream<float> L3_block1_conv2_bn_deno_stream;
    hls::stream<float> L3_block1_conv2_bn_gamma_stream;
    hls::stream<float> L3_block1_conv2_bn_beta_stream;

    //---------------------------------------------------------------

    // Dimension L4
    // L4 Block 0
    hls::stream<data_t> L4_block0_conv1_weights_stream;
    hls::stream<float> L4_block0_conv1_bn_mean_stream;
    hls::stream<float> L4_block0_conv1_bn_deno_stream;
    hls::stream<float> L4_block0_conv1_bn_gamma_stream;
    hls::stream<float> L4_block0_conv1_bn_beta_stream;

    hls::stream<data_t> L4_block0_conv2_weights_stream;
    hls::stream<float> L4_block0_conv2_bn_mean_stream;
    hls::stream<float> L4_block0_conv2_bn_deno_stream;
    hls::stream<float> L4_block0_conv2_bn_gamma_stream;
    hls::stream<float> L4_block0_conv2_bn_beta_stream;

    hls::stream<data_t> L4_skip_conv_weights_stream;
    hls::stream<float> L4_skip_bn_mean_stream;
    hls::stream<float> L4_skip_bn_deno_stream;
    hls::stream<float> L4_skip_bn_gamma_stream;
    hls::stream<float> L4_skip_bn_beta_stream;

    // L4 Block 1
    hls::stream<data_t> L4_block1_conv1_weights_stream;
    hls::stream<float> L4_block1_conv1_bn_mean_stream;
    hls::stream<float> L4_block1_conv1_bn_deno_stream;
    hls::stream<float> L4_block1_conv1_bn_gamma_stream;
    hls::stream<float> L4_block1_conv1_bn_beta_stream;

    hls::stream<data_t> L4_block1_conv2_weights_stream;
    hls::stream<float> L4_block1_conv2_bn_mean_stream;
    hls::stream<float> L4_block1_conv2_bn_deno_stream;
    hls::stream<float> L4_block1_conv2_bn_gamma_stream;
    hls::stream<float> L4_block1_conv2_bn_beta_stream;

    hls::stream<data_t> fc_weights_stream;
    hls::stream<data_t> fc_bias_stream;

    // ============================================================================
    // Load data from arrays into the streams
    // ============================================================================
    if (!load_array_to_stream(data_stream, input_block, INPUT_TENSOR_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }

    // L0
    if (!load_array_to_stream(L0_block_conv1_weights_stream, L0_block_conv1_weights, L0_BLOCK_CONV1_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L0_block_conv1_bn_mean_stream, L0_block_conv1_bn_mean, L0_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L0_block_conv1_bn_deno_stream, L0_block_conv1_bn_deno, L0_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L0_block_conv1_bn_gamma_stream, L0_block_conv1_bn_gamma, L0_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L0_block_conv1_bn_beta_stream, L0_block_conv1_bn_beta, L0_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    //-----------------------------------------------
    // L1 block 0 conv1
    if (!load_array_to_stream(L1_block0_conv1_weights_stream, L1_block0_conv1_weights, L1_BLOCK0_CONV1_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block0_conv1_bn_mean_stream, L1_block0_conv1_bn_mean, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block0_conv1_bn_deno_stream, L1_block0_conv1_bn_deno, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block0_conv1_bn_gamma_stream, L1_block0_conv1_bn_gamma, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block0_conv1_bn_beta_stream, L1_block0_conv1_bn_beta, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }

    // L1 block0 conv2
    if (!load_array_to_stream(L1_block0_conv2_weights_stream, L1_block0_conv2_weights, L1_BLOCK0_CONV2_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block0_conv2_bn_mean_stream, L1_block0_conv2_bn_mean, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block0_conv2_bn_deno_stream, L1_block0_conv2_bn_deno, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block0_conv2_bn_gamma_stream, L1_block0_conv2_bn_gamma, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block0_conv2_bn_beta_stream, L1_block0_conv2_bn_beta, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }

    // L1 block 1 conv1
    if (!load_array_to_stream(L1_block1_conv1_weights_stream, L1_block1_conv1_weights, L1_BLOCK1_CONV1_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block1_conv1_bn_mean_stream, L1_block1_conv1_bn_mean, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block1_conv1_bn_deno_stream, L1_block1_conv1_bn_deno, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block1_conv1_bn_gamma_stream, L1_block1_conv1_bn_gamma, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block1_conv1_bn_beta_stream, L1_block1_conv1_bn_beta, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    // L1 block1 conv2
    if (!load_array_to_stream(L1_block1_conv2_weights_stream, L1_block1_conv2_weights, L1_BLOCK1_CONV2_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block1_conv2_bn_mean_stream, L1_block1_conv2_bn_mean, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block1_conv2_bn_deno_stream, L1_block1_conv2_bn_deno, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block1_conv2_bn_gamma_stream, L1_block1_conv2_bn_gamma, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L1_block1_conv2_bn_beta_stream, L1_block1_conv2_bn_beta, L1_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    //-----------------------------------------------
    // L2 block 0 conv1
    if (!load_array_to_stream(L2_block0_conv1_weights_stream, L2_block0_conv1_weights, L2_BLOCK0_CONV1_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block0_conv1_bn_mean_stream, L2_block0_conv1_bn_mean, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block0_conv1_bn_deno_stream, L2_block0_conv1_bn_deno, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block0_conv1_bn_gamma_stream, L2_block0_conv1_bn_gamma, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block0_conv1_bn_beta_stream, L2_block0_conv1_bn_beta, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }

    // L2 block0 conv2
    if (!load_array_to_stream(L2_block0_conv2_weights_stream, L2_block0_conv2_weights, L2_BLOCK0_CONV2_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block0_conv2_bn_mean_stream, L2_block0_conv2_bn_mean, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block0_conv2_bn_deno_stream, L2_block0_conv2_bn_deno, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block0_conv2_bn_gamma_stream, L2_block0_conv2_bn_gamma, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block0_conv2_bn_beta_stream, L2_block0_conv2_bn_beta, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    // L2 Skip
    if (!load_array_to_stream(L2_skip_conv_weights_stream, L2_skip_conv_weights, L2_SKIP_CONV_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_skip_bn_mean_stream, L2_skip_bn_mean, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_skip_bn_deno_stream, L2_skip_bn_deno, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_skip_bn_gamma_stream, L2_skip_bn_gamma, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_skip_bn_beta_stream, L2_skip_bn_beta, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }

    // L2 block 1 conv1
    if (!load_array_to_stream(L2_block1_conv1_weights_stream, L2_block1_conv1_weights, L2_BLOCK1_CONV1_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block1_conv1_bn_mean_stream, L2_block1_conv1_bn_mean, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block1_conv1_bn_deno_stream, L2_block1_conv1_bn_deno, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block1_conv1_bn_gamma_stream, L2_block1_conv1_bn_gamma, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block1_conv1_bn_beta_stream, L2_block1_conv1_bn_beta, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    // L2 block1 conv2
    if (!load_array_to_stream(L2_block1_conv2_weights_stream, L2_block1_conv2_weights, L1_BLOCK1_CONV2_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block1_conv2_bn_mean_stream, L2_block1_conv2_bn_mean, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block1_conv2_bn_deno_stream, L2_block1_conv2_bn_deno, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block1_conv2_bn_gamma_stream, L2_block1_conv2_bn_gamma, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L2_block1_conv2_bn_beta_stream, L2_block1_conv2_bn_beta, L2_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    //-----------------------------------------------
    // L3 block 0 conv1
    if (!load_array_to_stream(L3_block0_conv1_weights_stream, L3_block0_conv1_weights, L3_BLOCK0_CONV1_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block0_conv1_bn_mean_stream, L3_block0_conv1_bn_mean, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block0_conv1_bn_deno_stream, L3_block0_conv1_bn_deno, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block0_conv1_bn_gamma_stream, L3_block0_conv1_bn_gamma, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block0_conv1_bn_beta_stream, L3_block0_conv1_bn_beta, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }

    // L3 block0 conv2
    if (!load_array_to_stream(L3_block0_conv2_weights_stream, L3_block0_conv2_weights, L3_BLOCK0_CONV2_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block0_conv2_bn_mean_stream, L3_block0_conv2_bn_mean, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block0_conv2_bn_deno_stream, L3_block0_conv2_bn_deno, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block0_conv2_bn_gamma_stream, L3_block0_conv2_bn_gamma, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block0_conv2_bn_beta_stream, L3_block0_conv2_bn_beta, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    // L3 Skip
    if (!load_array_to_stream(L3_skip_conv_weights_stream, L3_skip_conv_weights, L3_SKIP_CONV_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_skip_bn_mean_stream, L3_skip_bn_mean, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_skip_bn_deno_stream, L3_skip_bn_deno, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_skip_bn_gamma_stream, L3_skip_bn_gamma, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_skip_bn_beta_stream, L3_skip_bn_beta, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }

    // L3 block 1 conv1
    if (!load_array_to_stream(L3_block1_conv1_weights_stream, L3_block1_conv1_weights, L3_BLOCK1_CONV1_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block1_conv1_bn_mean_stream, L3_block1_conv1_bn_mean, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block1_conv1_bn_deno_stream, L3_block1_conv1_bn_deno, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block1_conv1_bn_gamma_stream, L3_block1_conv1_bn_gamma, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block1_conv1_bn_beta_stream, L3_block1_conv1_bn_beta, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    // L3 block1 conv2
    if (!load_array_to_stream(L3_block1_conv2_weights_stream, L3_block1_conv2_weights, L3_BLOCK1_CONV2_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block1_conv2_bn_mean_stream, L3_block1_conv2_bn_mean, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block1_conv2_bn_deno_stream, L3_block1_conv2_bn_deno, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block1_conv2_bn_gamma_stream, L3_block1_conv2_bn_gamma, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L3_block1_conv2_bn_beta_stream, L3_block1_conv2_bn_beta, L3_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    //-----------------------------------------------
    // L4 block 0 conv1
    if (!load_array_to_stream(L4_block0_conv1_weights_stream, L4_block0_conv1_weights, L4_BLOCK0_CONV1_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block0_conv1_bn_mean_stream, L4_block0_conv1_bn_mean, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block0_conv1_bn_deno_stream, L4_block0_conv1_bn_deno, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block0_conv1_bn_gamma_stream, L4_block0_conv1_bn_gamma, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block0_conv1_bn_beta_stream, L4_block0_conv1_bn_beta, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }

    // L4 block0 conv2
    if (!load_array_to_stream(L4_block0_conv2_weights_stream, L4_block0_conv2_weights, L4_BLOCK0_CONV2_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block0_conv2_bn_mean_stream, L4_block0_conv2_bn_mean, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block0_conv2_bn_deno_stream, L4_block0_conv2_bn_deno, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block0_conv2_bn_gamma_stream, L4_block0_conv2_bn_gamma, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block0_conv2_bn_beta_stream, L4_block0_conv2_bn_beta, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    // L4 Skip
    if (!load_array_to_stream(L4_skip_conv_weights_stream, L4_skip_conv_weights, L4_SKIP_CONV_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_skip_bn_mean_stream, L4_skip_bn_mean, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_skip_bn_deno_stream, L4_skip_bn_deno, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_skip_bn_gamma_stream, L4_skip_bn_gamma, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_skip_bn_beta_stream, L4_skip_bn_beta, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }

    // L4 block 1 conv1
    if (!load_array_to_stream(L4_block1_conv1_weights_stream, L4_block1_conv1_weights, L4_BLOCK1_CONV1_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block1_conv1_bn_mean_stream, L4_block1_conv1_bn_mean, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block1_conv1_bn_deno_stream, L4_block1_conv1_bn_deno, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block1_conv1_bn_gamma_stream, L4_block1_conv1_bn_gamma, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block1_conv1_bn_beta_stream, L4_block1_conv1_bn_beta, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    // L4 block1 conv2
    if (!load_array_to_stream(L4_block1_conv2_weights_stream, L4_block1_conv2_weights, L4_BLOCK1_CONV2_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block1_conv2_bn_mean_stream, L4_block1_conv2_bn_mean, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block1_conv2_bn_deno_stream, L4_block1_conv2_bn_deno, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block1_conv2_bn_gamma_stream, L4_block1_conv2_bn_gamma, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(L4_block1_conv2_bn_beta_stream, L4_block1_conv2_bn_beta, L4_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(fc_weights_stream, fc_weights, FC_WEIGHTS_SIZE))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }
    if (!load_array_to_stream(fc_bias_stream, fc_bias, FC_OUT_C))
    {
        std::cerr << "Error: Failed to load data into stream." << std::endl;
        return -1;
    }

    std::cout << "Data successfully loaded into hls::stream." << std::endl;

    // ============================================================================
    // Run the block
    // ============================================================================

    run_resnet_block(
        data_stream,
        output_stream,
        // L0
        L0_block_conv1_weights_stream,
        L0_block_conv1_bn_mean_stream,
        L0_block_conv1_bn_deno_stream,
        L0_block_conv1_bn_gamma_stream,
        L0_block_conv1_bn_beta_stream,
        
        //----------------------
        // L1 Block 0 conv1
        L1_block0_conv1_weights_stream,
        L1_block0_conv1_bn_mean_stream,
        L1_block0_conv1_bn_deno_stream,
        L1_block0_conv1_bn_gamma_stream,
        L1_block0_conv1_bn_beta_stream,
        // L1 Block 0 conv2
        L1_block0_conv2_weights_stream,
        L1_block0_conv2_bn_mean_stream,
        L1_block0_conv2_bn_deno_stream,
        L1_block0_conv2_bn_gamma_stream,
        L1_block0_conv2_bn_beta_stream,
        // L1 Block 1 conv1
        L1_block1_conv1_weights_stream,
        L1_block1_conv1_bn_mean_stream,
        L1_block1_conv1_bn_deno_stream,
        L1_block1_conv1_bn_gamma_stream,
        L1_block1_conv1_bn_beta_stream,
        // L1 Block1 conv2
        L1_block1_conv2_weights_stream,
        L1_block1_conv2_bn_mean_stream,
        L1_block1_conv2_bn_deno_stream,
        L1_block1_conv2_bn_gamma_stream,
        L1_block1_conv2_bn_beta_stream,
        
        //------------------
        // L2 Block 0 conv1
        L2_block0_conv1_weights_stream,
        L2_block0_conv1_bn_mean_stream,
        L2_block0_conv1_bn_deno_stream,
        L2_block0_conv1_bn_gamma_stream,
        L2_block0_conv1_bn_beta_stream,
        // L2 block0 conv2
        L2_block0_conv2_weights_stream,
        L2_block0_conv2_bn_mean_stream,
        L2_block0_conv2_bn_deno_stream,
        L2_block0_conv2_bn_gamma_stream,
        L2_block0_conv2_bn_beta_stream,
        // L2 Skip
        L2_skip_conv_weights_stream,
        L2_skip_bn_mean_stream,
        L2_skip_bn_deno_stream,
        L2_skip_bn_gamma_stream,
        L2_skip_bn_beta_stream,
        // L2 Block 1 conv1
        L2_block1_conv1_weights_stream,
        L2_block1_conv1_bn_mean_stream,
        L2_block1_conv1_bn_deno_stream,
        L2_block1_conv1_bn_gamma_stream,
        L2_block1_conv1_bn_beta_stream,
        // L2 block1 conv2
        L2_block1_conv2_weights_stream,
        L2_block1_conv2_bn_mean_stream,
        L2_block1_conv2_bn_deno_stream,
        L2_block1_conv2_bn_gamma_stream,
        L2_block1_conv2_bn_beta_stream,
        
        // ------------------
        // L3 Block 0 conv1
        L3_block0_conv1_weights_stream,
        L3_block0_conv1_bn_mean_stream,
        L3_block0_conv1_bn_deno_stream,
        L3_block0_conv1_bn_gamma_stream,
        L3_block0_conv1_bn_beta_stream,
        // L3 block 0 conv2
        L3_block0_conv2_weights_stream,
        L3_block0_conv2_bn_mean_stream,
        L3_block0_conv2_bn_deno_stream,
        L3_block0_conv2_bn_gamma_stream,
        L3_block0_conv2_bn_beta_stream,
        // L3 skip
        L3_skip_conv_weights_stream,
        L3_skip_bn_mean_stream,
        L3_skip_bn_deno_stream,
        L3_skip_bn_gamma_stream,
        L3_skip_bn_beta_stream,
        // L3 Block 1 conv1
        L3_block1_conv1_weights_stream,
        L3_block1_conv1_bn_mean_stream,
        L3_block1_conv1_bn_deno_stream,
        L3_block1_conv1_bn_gamma_stream,
        L3_block1_conv1_bn_beta_stream,
        // L3 block 1 conv2
        L3_block1_conv2_weights_stream,
        L3_block1_conv2_bn_mean_stream,
        L3_block1_conv2_bn_deno_stream,
        L3_block1_conv2_bn_gamma_stream,
        L3_block1_conv2_bn_beta_stream,
        
        // -----------------
        // L4 Block 0 conv1
        L4_block0_conv1_weights_stream,
        L4_block0_conv1_bn_mean_stream,
        L4_block0_conv1_bn_deno_stream,
        L4_block0_conv1_bn_gamma_stream,
        L4_block0_conv1_bn_beta_stream,
        // L4 block 0 conv2
        L4_block0_conv2_weights_stream,
        L4_block0_conv2_bn_mean_stream,
        L4_block0_conv2_bn_deno_stream,
        L4_block0_conv2_bn_gamma_stream,
        L4_block0_conv2_bn_beta_stream,
        // L4 Skip
        L4_skip_conv_weights_stream,
        L4_skip_bn_mean_stream,
        L4_skip_bn_deno_stream,
        L4_skip_bn_gamma_stream,
        L4_skip_bn_beta_stream,
        // L4 Block 1 conv1
        L4_block1_conv1_weights_stream,
        L4_block1_conv1_bn_mean_stream,
        L4_block1_conv1_bn_deno_stream,
        L4_block1_conv1_bn_gamma_stream,
        L4_block1_conv1_bn_beta_stream,
        // L4 block 1 conv2
        L4_block1_conv2_weights_stream,
        L4_block1_conv2_bn_mean_stream,
        L4_block1_conv2_bn_deno_stream,
        L4_block1_conv2_bn_gamma_stream,
        L4_block1_conv2_bn_beta_stream,
        fc_weights_stream,
        fc_bias_stream
        
    );
    printf("Block computation complete.\n");
    // printf("//I am here/// \n");

    // ============================================================================
    // Retrieve data from the output stream
    // ============================================================================
    for (int i = 0; i < OUTPUT_TENSOR_SIZE; i++)
    {
        if (!output_stream.empty())
        {
            output_block[i] = output_stream.read();
        }
        else
        {
            std::cerr << "Error: Output stream is empty at index " << i << std::endl;
            break;
        }
    }

    // ============================================================================
    // Print some portion of the output for verification
    // ============================================================================
    for (int i = 0; i < 10; i++)
    {
        std::cout << "Output tensor [" << i << "], " << output_block << std::endl;
    }

    // ============================================================================
    // Write out output tensor
    // ============================================================================
    ofstream outFile("output_tensor.txt");
    if (!outFile)
    {
        cerr << "Error: Could not open the file for writing." << endl;
        return 1;
    }
    for (int i = 0; i < OUTPUT_TENSOR_SIZE; i++)
    {
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
// ============================================================================
// Function to read data from a file into an array
// ============================================================================
template <typename T>
bool read_data_from_file(const char *filename, T *arr, int size)
{
    std::ifstream infile(filename);
    if (!infile.is_open())
    {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    for (int i = 0; i < size; ++i)
    {
        if (!(infile >> arr[i]))
        {
            std::cerr << "Error: Not enough data in file." << std::endl;
            return false;
        }
    }

    infile.close();
    return true;
}

// ============================================================================
// initializes all arrays by reading from text files
// ============================================================================
bool initialize_arrays()
{
    if (arrays_initialized)
    {
        return true; // Already initialized
    }

    bool success = true;

    success &= read_data_from_file("resnet18_input_data.txt", f_input_block, INPUT_TENSOR_SIZE);

    // L0
    success &= read_data_from_file("L0_block_conv1_weights.txt", f_L0_block_conv1_weights, L0_BLOCK_CONV1_SIZE);
    // success &= read_data_from_file("block_conv1_bias.txt", block_conv1_bias, BLOCK_CONV1_OUT_C);
    success &= read_data_from_file("L0_block_conv1_bn_mean.txt", L0_block_conv1_bn_mean, L0_OUT_C);
    success &= read_data_from_file("L0_block_conv1_bn_deno.txt", L0_block_conv1_bn_deno, L0_OUT_C);
    success &= read_data_from_file("L0_block_conv1_bn_gamma.txt", L0_block_conv1_bn_gamma, L0_OUT_C);
    success &= read_data_from_file("L0_block_conv1_bn_beta.txt", L0_block_conv1_bn_beta, L0_OUT_C);
    //-----------------------------------------------
    // L1 Block 0
    success &= read_data_from_file("L1_block0_conv1_weights.txt", f_L1_block0_conv1_weights, L1_BLOCK0_CONV1_SIZE);
    success &= read_data_from_file("L1_block0_conv1_bn_mean.txt", L1_block0_conv1_bn_mean, L1_OUT_C);
    success &= read_data_from_file("L1_block0_conv1_bn_deno.txt", L1_block0_conv1_bn_deno, L1_OUT_C);
    success &= read_data_from_file("L1_block0_conv1_bn_gamma.txt", L1_block0_conv1_bn_gamma, L1_OUT_C);
    success &= read_data_from_file("L1_block0_conv1_bn_beta.txt", L1_block0_conv1_bn_beta, L1_OUT_C);

    success &= read_data_from_file("L1_block0_conv2_weights.txt", f_L1__block0_conv2_weights, L1_BLOCK0_CONV2_SIZE);
    success &= read_data_from_file("L1_block0_conv2_bn_mean.txt", L1_block0_conv2_bn_mean, L1_OUT_C);
    success &= read_data_from_file("L1_block0_conv2_bn_deno.txt", L1_block0_conv2_bn_deno, L1_OUT_C);
    success &= read_data_from_file("L1_block0_conv2_bn_gamma.txt", L1_block0_conv2_bn_gamma, L1_OUT_C);
    success &= read_data_from_file("L1_block0_conv2_bn_beta.txt", L1_block0_conv2_bn_beta, L1_OUT_C);
    // L1 Block 1
    success &= read_data_from_file("L1_block1_conv1_weights.txt", f_L1_block1_conv1_weights, L1_BLOCK1_CONV1_SIZE);
    success &= read_data_from_file("L1_block1_conv1_bn_mean.txt", L1_block1_conv1_bn_mean, L1_OUT_C);
    success &= read_data_from_file("L1_block1_conv1_bn_deno.txt", L1_block1_conv1_bn_deno, L1_OUT_C);
    success &= read_data_from_file("L1_block1_conv1_bn_gamma.txt", L1_block1_conv1_bn_gamma, L1_OUT_C);
    success &= read_data_from_file("L1_block1_conv1_bn_beta.txt", L1_block1_conv1_bn_beta, L1_OUT_C);

    success &= read_data_from_file("L1_block1_conv2_weights.txt", f_L1_block1_conv2_weights, L1_BLOCK1_CONV2_SIZE);
    success &= read_data_from_file("L1_block1_conv2_bn_mean.txt", L1_block1_conv2_bn_mean, L1_OUT_C);
    success &= read_data_from_file("L1_block1_conv2_bn_deno.txt", L1_block1_conv2_bn_deno, L1_OUT_C);
    success &= read_data_from_file("L1_block1_conv2_bn_gamma.txt", L1_block1_conv2_bn_gamma, L1_OUT_C);
    success &= read_data_from_file("L1_block1_conv2_bn_beta.txt", L1_block1_conv2_bn_beta, L1_OUT_C);
    //---------------------------------------------------------
    // L2 Block 0
    success &= read_data_from_file("L2_block0_conv1_weights.txt", f_L2_block0_conv1_weights, L2_BLOCK0_CONV1_SIZE);
    success &= read_data_from_file("L2_block0_conv1_bn_mean.txt", L2_block0_conv1_bn_mean, L2_OUT_C);
    success &= read_data_from_file("L2_block0_conv1_bn_deno.txt", L2_block0_conv1_bn_deno, L2_OUT_C);
    success &= read_data_from_file("L2_block0_conv1_bn_gamma.txt", L2_block0_conv1_bn_gamma, L2_OUT_C);
    success &= read_data_from_file("L2_block0_conv1_bn_beta.txt", L2_block0_conv1_bn_beta, L2_OUT_C);

    success &= read_data_from_file("L2_block0_conv2_weights.txt", f_L2__block0_conv2_weights, L2_BLOCK0_CONV2_SIZE);
    success &= read_data_from_file("L2_block0_conv2_bn_mean.txt", L2_block0_conv2_bn_mean, L2_OUT_C);
    success &= read_data_from_file("L2_block0_conv2_bn_deno.txt", L2_block0_conv2_bn_deno, L2_OUT_C);
    success &= read_data_from_file("L2_block0_conv2_bn_gamma.txt", L2_block0_conv2_bn_gamma, L2_OUT_C);
    success &= read_data_from_file("L2_block0_conv2_bn_beta.txt", L2_block0_conv2_bn_beta, L2_OUT_C);

    // L2 Skip
    success &= read_data_from_file("L2_skip_conv_weights.txt", f_L2_skip_conv_weights, L2_SKIP_CONV_SIZE);
    // success &= read_data_from_file("skip_conv_bias.txt", skip_conv_bias, SKIP_CONV_OUT_C);
    success &= read_data_from_file("L2_skip_bn_mean.txt", L2_skip_bn_mean, L2_OUT_C);
    success &= read_data_from_file("L2_skip_bn_deno.txt", L2_skip_bn_deno, L2_OUT_C);
    success &= read_data_from_file("L2_skip_bn_gamma.txt", L2_skip_bn_gamma, L2_OUT_C);
    success &= read_data_from_file("L2_skip_bn_beta.txt", L2_skip_bn_beta, L2_OUT_C);

    // L2 Block 1
    success &= read_data_from_file("L2_block1_conv1_weights.txt", f_L2_block1_conv1_weights, L2_BLOCK1_CONV1_SIZE);
    success &= read_data_from_file("L2_block1_conv1_bn_mean.txt", L2_block1_conv1_bn_mean, L2_OUT_C);
    success &= read_data_from_file("L2_block1_conv1_bn_deno.txt", L2_block1_conv1_bn_deno, L2_OUT_C);
    success &= read_data_from_file("L2_block1_conv1_bn_gamma.txt", L2_block1_conv1_bn_gamma, L2_OUT_C);
    success &= read_data_from_file("L2_block1_conv1_bn_beta.txt", L2_block1_conv1_bn_beta, L2_OUT_C);

    success &= read_data_from_file("L2_block1_conv2_weights.txt", f_L2_block1_conv2_weights, L2_BLOCK1_CONV2_SIZE);
    success &= read_data_from_file("L2_block1_conv2_bn_mean.txt", L2_block1_conv2_bn_mean, L2_OUT_C);
    success &= read_data_from_file("L2_block1_conv2_bn_deno.txt", L2_block1_conv2_bn_deno, L2_OUT_C);
    success &= read_data_from_file("L2_block1_conv2_bn_gamma.txt", L2_block1_conv2_bn_gamma, L2_OUT_C);
    success &= read_data_from_file("L2_block1_conv2_bn_beta.txt", L2_block1_conv2_bn_beta, L2_OUT_C);

    //---------------------------------------------------------
    // L3 Block 0
    success &= read_data_from_file("L3_block0_conv1_weights.txt", f_L3_block0_conv1_weights, L3_BLOCK0_CONV1_SIZE);
    success &= read_data_from_file("L3_block0_conv1_bn_mean.txt", L3_block0_conv1_bn_mean, L3_OUT_C);
    success &= read_data_from_file("L3_block0_conv1_bn_deno.txt", L3_block0_conv1_bn_deno, L3_OUT_C);
    success &= read_data_from_file("L3_block0_conv1_bn_gamma.txt", L3_block0_conv1_bn_gamma, L3_OUT_C);
    success &= read_data_from_file("L3_block0_conv1_bn_beta.txt", L3_block0_conv1_bn_beta, L3_OUT_C);

    success &= read_data_from_file("L3_block0_conv2_weights.txt", f_L3__block0_conv2_weights, L3_BLOCK0_CONV2_SIZE);
    success &= read_data_from_file("L3_block0_conv2_bn_mean.txt", L3_block0_conv2_bn_mean, L3_OUT_C);
    success &= read_data_from_file("L3_block0_conv2_bn_deno.txt", L3_block0_conv2_bn_deno, L3_OUT_C);
    success &= read_data_from_file("L3_block0_conv2_bn_gamma.txt", L3_block0_conv2_bn_gamma, L3_OUT_C);
    success &= read_data_from_file("L3_block0_conv2_bn_beta.txt", L3_block0_conv2_bn_beta, L3_OUT_C);

    // L3 Skip
    success &= read_data_from_file("L3_skip_conv_weights.txt", f_L3_skip_conv_weights, L3_SKIP_CONV_SIZE);
    // success &= read_data_from_file("skip_conv_bias.txt", skip_conv_bias, SKIP_CONV_OUT_C);
    success &= read_data_from_file("L3_skip_bn_mean.txt", L3_skip_bn_mean, L3_OUT_C);
    success &= read_data_from_file("L3_skip_bn_deno.txt", L3_skip_bn_deno, L3_OUT_C);
    success &= read_data_from_file("L3_skip_bn_gamma.txt", L3_skip_bn_gamma, L3_OUT_C);
    success &= read_data_from_file("L3_skip_bn_beta.txt", L3_skip_bn_beta, L3_OUT_C);

    // L3 Block 1
    success &= read_data_from_file("L3_block1_conv1_weights.txt", f_L3_block1_conv1_weights, L3_BLOCK1_CONV1_SIZE);
    success &= read_data_from_file("L3_block1_conv1_bn_mean.txt", L3_block1_conv1_bn_mean, L3_OUT_C);
    success &= read_data_from_file("L3_block1_conv1_bn_deno.txt", L3_block1_conv1_bn_deno, L3_OUT_C);
    success &= read_data_from_file("L3_block1_conv1_bn_gamma.txt", L3_block1_conv1_bn_gamma, L3_OUT_C);
    success &= read_data_from_file("L3_block1_conv1_bn_beta.txt", L3_block1_conv1_bn_beta, L3_OUT_C);

    success &= read_data_from_file("L3_block1_conv2_weights.txt", f_L3_block1_conv2_weights, L3_BLOCK1_CONV2_SIZE);
    success &= read_data_from_file("L3_block1_conv2_bn_mean.txt", L3_block1_conv2_bn_mean, L3_OUT_C);
    success &= read_data_from_file("L3_block1_conv2_bn_deno.txt", L3_block1_conv2_bn_deno, L3_OUT_C);
    success &= read_data_from_file("L3_block1_conv2_bn_gamma.txt", L3_block1_conv2_bn_gamma, L3_OUT_C);
    success &= read_data_from_file("L3_block1_conv2_bn_beta.txt", L3_block1_conv2_bn_beta, L3_OUT_C);

    //---------------------------------------------------------
    // L4 Block 0
    success &= read_data_from_file("L4_block0_conv1_weights.txt", f_L4_block0_conv1_weights, L4_BLOCK0_CONV1_SIZE);
    success &= read_data_from_file("L4_block0_conv1_bn_mean.txt", L4_block0_conv1_bn_mean, L4_OUT_C);
    success &= read_data_from_file("L4_block0_conv1_bn_deno.txt", L4_block0_conv1_bn_deno, L4_OUT_C);
    success &= read_data_from_file("L4_block0_conv1_bn_gamma.txt", L4_block0_conv1_bn_gamma, L4_OUT_C);
    success &= read_data_from_file("L4_block0_conv1_bn_beta.txt", L4_block0_conv1_bn_beta, L4_OUT_C);

    success &= read_data_from_file("L4_block0_conv2_weights.txt", f_L4__block0_conv2_weights, L4_BLOCK0_CONV2_SIZE);
    success &= read_data_from_file("L4_block0_conv2_bn_mean.txt", L4_block0_conv2_bn_mean, L4_OUT_C);
    success &= read_data_from_file("L4_block0_conv2_bn_deno.txt", L4_block0_conv2_bn_deno, L4_OUT_C);
    success &= read_data_from_file("L4_block0_conv2_bn_gamma.txt", L4_block0_conv2_bn_gamma, L4_OUT_C);
    success &= read_data_from_file("L4_block0_conv2_bn_beta.txt", L4_block0_conv2_bn_beta, L4_OUT_C);

    // L4 Skip
    success &= read_data_from_file("L4_skip_conv_weights.txt", f_L4_skip_conv_weights, L4_SKIP_CONV_SIZE);
    // success &= read_data_from_file("skip_conv_bias.txt", skip_conv_bias, SKIP_CONV_OUT_C);
    success &= read_data_from_file("L4_skip_bn_mean.txt", L4_skip_bn_mean, L4_OUT_C);
    success &= read_data_from_file("L4_skip_bn_deno.txt", L4_skip_bn_deno, L4_OUT_C);
    success &= read_data_from_file("L4_skip_bn_gamma.txt", L4_skip_bn_gamma, L4_OUT_C);
    success &= read_data_from_file("L4_skip_bn_beta.txt", L4_skip_bn_beta, L4_OUT_C);

    // L4 Block 1
    success &= read_data_from_file("L4_block1_conv1_weights.txt", f_L4_block1_conv1_weights, L4_BLOCK1_CONV1_SIZE);
    success &= read_data_from_file("L4_block1_conv1_bn_mean.txt", L4_block1_conv1_bn_mean, L4_OUT_C);
    success &= read_data_from_file("L4_block1_conv1_bn_deno.txt", L4_block1_conv1_bn_deno, L4_OUT_C);
    success &= read_data_from_file("L4_block1_conv1_bn_gamma.txt", L4_block1_conv1_bn_gamma, L4_OUT_C);
    success &= read_data_from_file("L4_block1_conv1_bn_beta.txt", L4_block1_conv1_bn_beta, L4_OUT_C);

    success &= read_data_from_file("L4_block1_conv2_weights.txt", f_L4_block1_conv2_weights, L4_BLOCK1_CONV2_SIZE);
    success &= read_data_from_file("L4_block1_conv2_bn_mean.txt", L4_block1_conv2_bn_mean, L4_OUT_C);
    success &= read_data_from_file("L4_block1_conv2_bn_deno.txt", L4_block1_conv2_bn_deno, L4_OUT_C);
    success &= read_data_from_file("L4_block1_conv2_bn_gamma.txt", L4_block1_conv2_bn_gamma, L4_OUT_C);
    success &= read_data_from_file("L4_block1_conv2_bn_beta.txt", L4_block1_conv2_bn_beta, L4_OUT_C);

    // fc
    success &= read_data_from_file("fc_weights.txt", f_fc_weights, FC_WEIGHTS_SIZE);
    success &= read_data_from_file("fc_bias.txt", f_fc_bias, FC_OUT_C);

    if (!success)
    {
        std::cerr << "Error: Failed to load one or more weight/BN files." << std::endl;
        return false;
    }

    arrays_initialized = true;
    return true;
}
