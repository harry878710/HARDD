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
typedef ap_fixed<8,4>     data_t;   // 4 bits integer, 4 bits fractional

// For accumulations to prevent overflow
typedef ap_fixed<16,8>    acc_t;    // 8 bits integer, 8 bits fractional

// Dimensions for this block (layer3)
#define IN_H 14
#define IN_W 14
#define IN_C 256
#define OUT_C 512
#define INPUT_TENSOR_SIZE IN_H*IN_W*IN_C
#define OUTPUT_TENSOR_SIZE (IN_H/2)*(IN_W/2)*OUT_C

//Dimension for layer0
#define layer0_out_h 112
#define layer0_out_w 112
#define layer0_out_ch 64

//Dimension for layer1
#define layer1_out_h 56
#define layer1_out_w 56
#define layer1_out_ch 64

//Dimension for layer2
#define layer2_out_h 28
#define layer2_out_w 28
#define layer2_out_ch 128
#define LAYER2_SIZE = 56 * 56 * 64; // Adjust as needed

//Dimension for layer3
#define layer3_out_h 14
#define layer3_out_w 14
#define layer3_out_ch 256
#define LAYER3_SIZE = 28 * 28 * 128; // Adjust as needed

//Dimension for layer4
#define layer4_out_h 7
#define layer4_out_w 7
#define layer4_out_ch 512
#define LAYER4_SIZE = 14 * 14 * 256; // Adjust as needed

// Kernel sizes
#define K 3

// For skip connection convolution
#define SKIP_K 1

// For set up input 1-D Arrays
#define BLOCK_CONV1_OUT_C 512
#define BLOCK_CONV1_IN_C 256
#define BLOCK_CONV1_SIZE (BLOCK_CONV1_OUT_C * BLOCK_CONV1_IN_C * K * K)

#define BLOCK_CONV2_OUT_C 512
#define BLOCK_CONV2_IN_C 512
#define BLOCK_CONV2_SIZE (BLOCK_CONV2_OUT_C * BLOCK_CONV2_IN_C * K * K)

#define SKIP_CONV_OUT_C 512
#define SKIP_CONV_IN_C 256
#define SKIP_CONV_SIZE (SKIP_CONV_OUT_C * SKIP_CONV_IN_C * SKIP_K * SKIP_K)

// Quantization scale parameter
static const float IN_SCALE = 16.0f;
static const float OUT_SCALE = 16.0f;

#endif
