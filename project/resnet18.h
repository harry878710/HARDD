// resnet18.h
// include all header files in this header file
#ifndef RESNET18_H
#define RESNET18_H

#include <ap_fixed.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <hls_math.h>
#include <hls_stream.h>
#include <cstdlib>
#include <cstdint>
#include <string>
// #include <vector>

// ============================================================================
// Data Type Definitions
// ============================================================================

// Fixed-point data type for inputs, weights, and outputs
typedef ap_fixed<8,4, AP_RND, AP_SAT>     data_t;   // <total-bit, int-bit>   4 bits integer, 4 bits fractional

// For accumulations to prevent overflow
typedef ap_fixed<32,16, AP_RND, AP_SAT>    acc_t;    // 8 bits integer, 8 bits fractional

// Dimensions for this block
// #define IN_H 14
// #define IN_W 14
// #define IN_C 256
// #define OUT_C 512
// #define INPUT_TENSOR_SIZE IN_H*IN_W*IN_C
// #define OUTPUT_TENSOR_SIZE (IN_H/2)*(IN_W/2)*OUT_C

// Kernel sizes
#define K 3

// For skip connection convolution
#define SKIP_K 1

// For set up input 1-D Arrays
// #define BLOCK_CONV1_OUT_C 512
// #define BLOCK_CONV1_IN_C 256
// #define BLOCK_CONV1_SIZE (BLOCK_CONV1_OUT_C * BLOCK_CONV1_IN_C * K * K)

// #define BLOCK_CONV2_OUT_C 512
// #define BLOCK_CONV2_IN_C 512
// #define BLOCK_CONV2_SIZE (BLOCK_CONV2_OUT_C * BLOCK_CONV2_IN_C * K * K)

// #define SKIP_CONV_OUT_C 512
// #define SKIP_CONV_IN_C 256
// #define SKIP_CONV_SIZE (SKIP_CONV_OUT_C * SKIP_CONV_IN_C * SKIP_K * SKIP_K)

// Input Block
#define INPUT_TENSOR_SIZE 224 * 224 * 3
#define OUTPUT_TENSOR_SIZE 1 * 1 * 1000

// Dimensions L0
#define L0_IN_H 224
#define L0_IN_W 224
#define L0_IN_C 3
#define L0_OUT_C 64
#define L0_INPUT_TENSOR_SIZE L0_IN_H *L0_IN_W *L0_IN_C
#define L0_OUTPUT_TENSOR_SIZE (L0_IN_H / 2) * (L0_IN_W / 2) * L0_OUT_C

// Kernel sizes
#define L0_K 7

// For L0 Block
#define L0_BLOCK_CONV1_IN_C 3
#define L0_BLOCK_CONV1_OUT_C 64
#define L0_BLOCK_CONV1_SIZE (L0_BLOCK_CONV1_IN_C * L0_BLOCK_CONV1_OUT_C * L0_K * L0_K)
//-----------------------------------------

// Dimensions L1
#define L1_IN_H 112
#define L1_IN_W 112
#define L1_IN_C 64
#define L1_OUT_C 64
#define L1_INPUT_TENSOR_SIZE L1_IN_H *L1_IN_W *L1_IN_C
#define L1_OUTPUT_TENSOR_SIZE (L1_IN_H / 2) * (L1_IN_W / 2) * L1_OUT_C

// Kernel sizes
#define L1_K 3

// For L1 Block0
#define L1_BLOCK0_CONV1_IN_C 64
#define L1_BLOCK0_CONV1_OUT_C 64
#define L1_BLOCK0_CONV1_SIZE (L1_BLOCK0_CONV1_IN_C * L1_BLOCK0_CONV1_OUT_C * L1_K * L1_K)

#define L1_BLOCK0_CONV2_IN_C 64
#define L1_BLOCK0_CONV2_OUT_C 64
#define L1_BLOCK0_CONV2_SIZE (L1_BLOCK0_CONV2_IN_C * L1_BLOCK0_CONV2_OUT_C * L1_K * L1_K)

// For L1 Block1
#define L1_BLOCK1_CONV1_IN_C 64
#define L1_BLOCK1_CONV1_OUT_C 64
#define L1_BLOCK1_CONV1_SIZE (L1_BLOCK1_CONV1_IN_C * L1_BLOCK1_CONV1_OUT_C * L1_K * L1_K)

// For L1 Block1
#define L1_BLOCK1_CONV2_IN_C 64
#define L1_BLOCK1_CONV2_OUT_C 64
#define L1_BLOCK1_CONV2_SIZE (L1_BLOCK1_CONV2_IN_C * L1_BLOCK1_CONV2_OUT_C * L1_K * L1_K)
//-----------------------------------------

// Dimensions L2
#define L2_IN_H 56
#define L2_IN_W 56
#define L2_IN_C 64
#define L2_OUT_C 128
#define L2_INPUT_TENSOR_SIZE L2_IN_H *L2_IN_W *L2_IN_C
#define L2_OUTPUT_TENSOR_SIZE (L2_IN_H / 2) * (L2_IN_W / 2) * L2_OUT_C

// Kernel sizes
#define L2_K 3

// For skip connection convolution
#define L2_SKIP_K 1

// For L2 Block0
#define L2_BLOCK0_CONV1_IN_C 64
#define L2_BLOCK0_CONV1_OUT_C 128
#define L2_BLOCK0_CONV1_SIZE (L2_BLOCK0_CONV1_IN_C * L2_BLOCK0_CONV1_OUT_C * L2_K * L2_K)

#define L2_BLOCK0_CONV2_IN_C 128
#define L2_BLOCK0_CONV2_OUT_C 128
#define L2_BLOCK0_CONV2_SIZE (L2_BLOCK0_CONV2_IN_C * L2_BLOCK0_CONV2_OUT_C * L2_K * L2_K)

// For L2 SKIP
#define L2_SKIP_CONV_IN_C 64
#define L2_SKIP_CONV_OUT_C 128
#define L2_SKIP_CONV_SIZE (L2_SKIP_CONV_IN_C * L2_SKIP_CONV_OUT_C * L2_SKIP_K * L2_SKIP_K)

// For L2 Block1
#define L2_BLOCK1_CONV1_IN_C 128
#define L2_BLOCK1_CONV1_OUT_C 128
#define L2_BLOCK1_CONV1_SIZE (L2_BLOCK1_CONV1_IN_C * L2_BLOCK1_CONV1_OUT_C * L2_K * L2_K)

// For L2 Block1
#define L2_BLOCK1_CONV2_IN_C 128
#define L2_BLOCK1_CONV2_OUT_C 128
#define L2_BLOCK1_CONV2_SIZE (L2_BLOCK1_CONV2_IN_C * L2_BLOCK1_CONV2_OUT_C * L2_K * L2_K)
//-----------------------------------------
// Dimensions L3
#define L3_IN_H 28
#define L3_IN_W 28
#define L3_IN_C 128
#define L3_OUT_C 256
#define L3_INPUT_TENSOR_SIZE L3_IN_H *L3_IN_W *L3_IN_C
#define L3_OUTPUT_TENSOR_SIZE (L3_IN_H / 2) * (L3_IN_W / 2) * L3_OUT_C

// Kernel sizes
#define L3_K 3

// For skip connection convolution
#define L3_SKIP_K 1

// For L3 Block0
#define L3_BLOCK0_CONV1_IN_C 128
#define L3_BLOCK0_CONV1_OUT_C 256
#define L3_BLOCK0_CONV1_SIZE (L3_BLOCK0_CONV1_IN_C * L3_BLOCK0_CONV1_OUT_C * L3_K * L3_K)

#define L3_BLOCK0_CONV2_IN_C 256
#define L3_BLOCK0_CONV2_OUT_C 256
#define L3_BLOCK0_CONV2_SIZE (L3_BLOCK0_CONV2_IN_C * L3_BLOCK0_CONV2_OUT_C * L3_K * L3_K)

// For L3 SKIP
#define L3_SKIP_CONV_IN_C 128
#define L3_SKIP_CONV_OUT_C 256
#define L3_SKIP_CONV_SIZE (L3_SKIP_CONV_IN_C * L3_SKIP_CONV_OUT_C * L3_SKIP_K * L3_SKIP_K)

// For L3 Block1
#define L3_BLOCK1_CONV1_IN_C 256
#define L3_BLOCK1_CONV1_OUT_C 256
#define L3_BLOCK1_CONV1_SIZE (L3_BLOCK1_CONV1_IN_C * L3_BLOCK1_CONV1_OUT_C * L3_K * L3_K)

// For L3 Block1
#define L3_BLOCK1_CONV2_IN_C 256
#define L3_BLOCK1_CONV2_OUT_C 256
#define L3_BLOCK1_CONV2_SIZE (L3_BLOCK1_CONV2_IN_C * L3_BLOCK1_CONV2_OUT_C * L3_K * L3_K)

// -----------------------------------
// Dimensions L4
#define L4_IN_H 14
#define L4_IN_W 14
#define L4_IN_C 256
#define L4_OUT_C 512
#define L4_INPUT_TENSOR_SIZE L4_IN_H *L4_IN_W *L4_IN_C
#define L4_OUTPUT_TENSOR_SIZE (L4_IN_H / 2) * (L4_IN_W / 2) * L4_OUT_C

// Kernel sizes
#define L4_K 3

// For skip connection convolution
#define L4_SKIP_K 1

// For L4 Block0
#define L4_BLOCK0_CONV1_IN_C 256
#define L4_BLOCK0_CONV1_OUT_C 512
#define L4_BLOCK0_CONV1_SIZE (L4_BLOCK0_CONV1_OUT_C * L4_BLOCK0_CONV1_IN_C * L4_K * L4_K)

#define L4_BLOCK0_CONV2_IN_C 512
#define L4_BLOCK0_CONV2_OUT_C 512
#define L4_BLOCK0_CONV2_SIZE (L4_BLOCK0_CONV2_OUT_C * L4_BLOCK0_CONV2_IN_C * L4_K * L4_K)

// For L4 SKIP
#define L4_SKIP_CONV_IN_C 256
#define L4_SKIP_CONV_OUT_C 512
#define L4_SKIP_CONV_SIZE (L4_SKIP_CONV_IN_C * L4_SKIP_CONV_OUT_C * L4_SKIP_K * L4_SKIP_K)

// For L4 Block1
#define L4_BLOCK1_CONV1_IN_C 512
#define L4_BLOCK1_CONV1_OUT_C 512
#define L4_BLOCK1_CONV1_SIZE (L4_BLOCK1_CONV1_IN_C * L4_BLOCK1_CONV1_OUT_C * L4_K * L4_K)

// For L4 Block1
#define L4_BLOCK1_CONV2_IN_C 512
#define L4_BLOCK1_CONV2_OUT_C 512
#define L4_BLOCK1_CONV2_SIZE (L4_BLOCK1_CONV2_IN_C * L4_BLOCK1_CONV2_OUT_C * L4_K * L4_K)

// fc
#define FC_IN_C 512
#define FC_OUT_C 1000
#define FC_WEIGHTS_SIZE (FC_IN_C*FC_OUT_C)

// --------------------------------------

// Quantization scale parameter
static const float DATA_SCALE = 16.0f; // Change based on the ap_fixed precision
static const float ACC_SCALE = 65536.0f; // 2^(16) Change based on the ap_fixed precision 
static const int MAX_DATA_T = 127;  // Change based on the ap_fixed precision
static const int MIN_DATA_T = -128; // Change based on the ap_fixed precision

#endif
