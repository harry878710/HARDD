// resnet18.h
// include all header files in this header file
#ifndef RESNET18_H
#define RESNET18_H

#include <ap_fixed.h>
#include <hls_stream.h>
#include <iostream>
#include <cmath>
#include <hls_math.h>
#include <BN.h>
#include <ReLU.h>
#include <skipConnect.h>
#include <quantize.h>
#include <block_function.h>

// ============================================================================
// Data Type Definitions
// ============================================================================

// Fixed-point data type for inputs, weights, and outputs
typedef ap_fixed<8,4>     data_t;   // 4 bits integer, 4 bits fractional

// For accumulations to prevent overflow
typedef ap_fixed<32,16>    acc_t;    // 16 bits integer, 16 bits fractional

// Dimensions for this block
#define IN_H 14
#define IN_W 14
#define IN_C 256
#define OUT_C 512

// Kernel sizes
#define K 3

// For skip connection convolution
#define SKIP_K 1

#endif


