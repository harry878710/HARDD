// conv.cpp
#include "conv.h"
template void conv2D<512, 256, 14, 14, 3>(hls::stream<data_t>&, hls::stream<data_t>&, hls::stream<data_t>&, int, int);
template void conv2D<512, 512, 7, 7, 3>(hls::stream<data_t>&, hls::stream<data_t>&, hls::stream<data_t>&, int, int);
template void conv2D<512, 256, 14, 14, 1>(hls::stream<data_t>&, hls::stream<data_t>&, hls::stream<data_t>&, int, int);

template<int CONV_OUT_C, int CONV_IN_C, int CONV_IN_H, int CONV_IN_W, int CONV_K>
void conv2D(
    hls::stream<data_t> &input,
    hls::stream<data_t> &output,
    hls::stream<data_t> &weights,
    int stride, int pad
) {
    int out_h = CONV_IN_H / stride;
    int out_w = CONV_IN_W / stride;

    // data_t weight_buffer[OUT_C][IN_C][K][K];
    data_t weight_buffer[CONV_OUT_C][CONV_IN_C][CONV_K][CONV_K];
    #pragma HLS ARRAY_PARTITION variable=weight_buffer dim=2 factor=4 cyclic
    #pragma HLS BIND_STORAGE variable=weight_buffer type=ram_2p impl=uram
    // Load weights
    for(int oc = 0; oc < CONV_OUT_C; oc++) {
        #pragma HLS DATAFLOW
        for(int ic = 0; ic < CONV_IN_C; ic++) {
            for(int kh = 0; kh < CONV_K; kh++) {
                for(int kw = 0; kw < CONV_K; kw++) {
                    #pragma HLS PIPELINE II=1
                    // #pragma HLS UNROLL factor=4
                    weight_buffer[oc][ic][kh][kw] = weights.read();
                }
            }
        }
    }
    
    // data_t input_buffer[IN_C][CONV_IN_H][CONV_IN_W];
    data_t input_buffer[CONV_IN_C][CONV_IN_H][CONV_IN_W];
    #pragma HLS ARRAY_PARTITION variable=input_buffer dim=1 factor=4 cyclic
    #pragma HLS BIND_STORAGE variable=input_buffer type=ram_2p impl=bram

    // Load input buffer
    for(int ic = 0; ic < CONV_IN_C; ic++) {
        #pragma HLS DATAFLOW
        for(int ih = 0; ih < CONV_IN_H; ih++) {
            for(int iw = 0; iw < CONV_IN_W; iw++) {
                // #pragma HLS UNROLL
                #pragma HLS PIPELINE II=1
                input_buffer[ic][ih][iw] = input.read();
            }
        }
    }

    for (int oc = 0; oc < CONV_OUT_C; oc++) {
        #pragma HLS DATAFLOW
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                // acc_t acc = bias[oc]; // bias is in acc type, which is 32-bit
                acc_t acc = 0;
                for (int kh = 0; kh < CONV_K; kh++) {
                    for (int kw = 0; kw < CONV_K; kw++) {
                        int ih = oh*stride + kh - pad;
                        int iw = ow*stride + kw - pad;
                        if (ih >= 0 && ih < CONV_IN_H && iw >= 0 && iw < CONV_IN_W) {
                            for (int ic = 0; ic < CONV_IN_C; ic++) {
                                #pragma HLS PIPELINE II=1
                                acc_t val = (acc_t)input_buffer[ic][ih][iw];
                                acc_t w = (acc_t)weight_buffer[oc][ic][kh][kw];
                                acc += val * w;
                            }
                        }
                    }
                }
                data_t q_val = quantize_acc_to_data(acc);
                output.write(q_val);
            }
        }
    }
}
