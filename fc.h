// fc.h
#ifndef FC_H
#define FC_H
#include "resnet18.h"

// Fully Connected Layer Function
void fully_connected(
    hls::stream<data_t> &in_stream,        // Input vector
    hls::stream<data_t> &out_stream,       // Output vector
    data_t weights[1000][512],             // Weight matrix [out_ch][in_ch]
    data_t biases[1000],                   // Bias vector [out_ch]
    int in_ch = 512,
    int out_ch = 1000
);
#endif
