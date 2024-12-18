// fc.h
#ifndef FC_H
#define FC_H
#include "resnet18.h"

// Fully Connected Layer Function
void fully_connected(
    hls::stream<data_t> &in_stream,        // Input vector
    hls::stream<data_t> &out_stream,       // Output vector
    hls::stream<data_t> &weights,             // Weight matrix [out_ch][in_ch]
    hls::stream<data_t> &biases,                   // Bias vector [out_ch]
    int in_ch = 512,
    int out_ch = 1000
);
#endif
