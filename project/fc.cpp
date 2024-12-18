// fc.cpp
#include "fc.h"

// Fully Connected Layer Function
void fully_connected(
    hls::stream<data_t> &in_stream,        // Input vector
    hls::stream<data_t> &out_stream,       // Output vector
    hls::stream<data_t> &weights,             // Weight matrix [out_ch][in_ch]
    hls::stream<data_t> &biases,                   // Bias vector [out_ch]
    int in_ch = 512,
    int out_ch = 1000
) {
    // Iterate over each output neuron
    for(int j = 0; j < out_ch; j++) {  // 1000
        #pragma HLS PIPELINE II=1
        acc_t sum = 0.0;
        // Iterate over each input dimension
        for(int i = 0; i < in_ch; i++) {  // 512
            #pragma HLS PIPELINE II=1
            // #pragma HLS UNROLL factor=4
            data_t in_val = in_stream.read();
            sum += (acc_t)weights[j][i] * (acc_t)in_val;
        }
        sum += (acc_t)biases[j];
        data_t d_sum = clamp_acc_to_data(sum);
        out_stream.write(d_sum);
    }
}
