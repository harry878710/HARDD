// conv.cpp
#include "conv.h"

void conv2D(
    hls::stream<data_t> &input,
    hls::stream<data_t> &output,
    hls::stream<data_t> &weights,
    int in_h, int in_w, int in_ch,
    int out_ch, int k_size, int stride, int pad
) {
    // Calculate output dimensions
    int out_h = (in_h + 2 * pad - k_size) / stride + 1;
    int out_w = (in_w + 2 * pad - k_size) / stride + 1;

    // Load weights into local buffer
    data_t weight_buffer[out_ch][in_ch][k_size][k_size];
// #pragma HLS ARRAY_PARTITION variable=weight_buffer complete dim=1
// #pragma HLS ARRAY_PARTITION variable=weight_buffer complete dim=2
    for(int oc = 0; oc < out_ch; oc++) {
        for(int ic = 0; ic < in_ch; ic++) {
            for(int kh = 0; kh < k_size; kh++) {
                for(int kw = 0; kw < k_size; kw++) {
#pragma HLS PIPELINE II=1
                    weight_buffer[oc][ic][kh][kw] = weights.read();
                }
            }
        }
    }

    // Initialize sliding window buffers
    data_t window_buffer[in_ch][k_size][k_size];
// #pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=1
    for(int ic = 0; ic < in_ch; ic++) {
        for(int kh = 0; kh < k_size; kh++) {
            for(int kw = 0; kw < k_size; kw++) {
#pragma HLS UNROLL
                window_buffer[ic][kh][kw] = 0;
            }
        }
    }

    // Perform convolution using sliding window
    for(int ih = 0; ih < in_h + 2 * pad; ih++) {
        for(int iw = 0; iw < in_w + 2 * pad; iw++) {
#pragma HLS PIPELINE II=1
            // Read input or apply zero-padding
            data_t input_val;
            if(ih < pad || ih >= in_h + pad || iw < pad || iw >= in_w + pad) {
                input_val = 0; // Zero-padding
            }
            else {
                input_val = input.read();
            }

            // Shift window buffer
            for(int ic = 0; ic < in_ch; ic++) {
#pragma HLS UNROLL
                for(int kh = k_size - 1; kh > 0; kh--) {
#pragma HLS UNROLL
                    for(int kw = k_size - 1; kw > 0; kw--) {
#pragma HLS UNROLL
                        window_buffer[ic][kh][kw] = window_buffer[ic][kh][kw - 1];
                    }
                    window_buffer[ic][kh][0] = window_buffer[ic][kh - 1][k_size - 1];
                }
                window_buffer[ic][0][0] = input_val;
            }

            // Perform convolution if within output bounds
            if((ih >= pad) && ((ih - pad) % stride == 0) && (ih - pad) / stride < out_h &&
               (iw >= pad) && ((iw - pad) % stride == 0) && (iw - pad) / stride < out_w) {
                int oh = (ih - pad) / stride;
                int ow = (iw - pad) / stride;

                for(int oc = 0; oc < out_ch; oc++) {
#pragma HLS UNROLL
                    acc_t acc = 0;
                    for(int ic = 0; ic < in_ch; ic++) {
#pragma HLS UNROLL
                        for(int kh = 0; kh < k_size; kh++) {
#pragma HLS UNROLL
                            for(int kw = 0; kw < k_size; kw++) {
#pragma HLS UNROLL
                                acc += window_buffer[ic][kh][kw] * weight_buffer[oc][ic][kh][kw];
                            }
                        }
                    }
                    // Quantize and write to output stream
                    data_t q_val = quantize_acc_to_data(acc);
                    output.write(q_val);
                }
            }
        }
    }
}
