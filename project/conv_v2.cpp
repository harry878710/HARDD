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
    // Consider partial partitioning along 'ic' and 'oc' to allow parallel reads.
    data_t weight_buffer[OUT_C][IN_C][K][K];
    // #pragma HLS ARRAY_PARTITION variable=weight_buffer dim=2 factor=4 cyclic
    // #pragma HLS BIND_STORAGE variable=weight_buffer type=ram_2p impl=bram
    // The above chooses a partial partition for input channels by a factor of 4.
    // Adjust the factor based on resource and performance trade-offs.
    
    // Load weights
    // load_weights:
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

    data_t window_buffer[IN_C][K][K];
    // #pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=1
    // #pragma HLS BIND_STORAGE variable=window_buffer type=ram_1p impl=bram

    // Initialize window buffer
    // init_window:
    for(int ic = 0; ic < in_ch; ic++) {
        for(int kh = 0; kh < k_size; kh++) {
            for(int kw = 0; kw < k_size; kw++) {
                // #pragma HLS UNROLL
                window_buffer[ic][kh][kw] = (data_t)0;
            }
        }
    }

    // Input reading loops
    // input_loop:
    for(int ih = 0; ih < in_h + 2 * pad; ih++) {
        for(int iw = 0; iw < in_w + 2 * pad; iw++) {
            #pragma HLS PIPELINE II=1
            data_t input_val = (data_t)0;
            if(ih >= pad && ih < in_h + pad && iw >= pad && iw < in_w + pad) {
                input_val = input.read();
            }

            // Shift window buffer
            // A shift register approach could be used, but it is already done here.
            // Optimization: Since we read one pixel at a time, we can shift directly.
            // If too complex, consider using a line buffer or a simpler shifting logic.
            for(int ic = 0; ic < in_ch; ic++) {
                #pragma HLS UNROLL factor=4
                // Shift window
                for(int kh = k_size - 1; kh > 0; kh--) {
                    for(int kw = k_size - 1; kw > 0; kw--) {
                        // #pragma HLS UNROLL
                        window_buffer[ic][kh][kw] = window_buffer[ic][kh][kw-1];
                    }
                    window_buffer[ic][kh][0] = window_buffer[ic][kh-1][k_size-1];
                }
                window_buffer[ic][0][0] = input_val;
            }

            // Perform convolution if within output bounds
            if((ih >= pad) && ((ih - pad) % stride == 0) && ((ih - pad)/stride < out_h) &&
               (iw >= pad) && ((iw - pad) % stride == 0) && ((iw - pad)/stride < out_w)) {
                int oh = (ih - pad) / stride;
                int ow = (iw - pad) / stride;

                // output_out_ch:
                for(int oc = 0; oc < out_ch; oc++) {
                    #pragma HLS PIPELINE II=1
                    acc_t acc = 0;
                    // compute_ic:
                    for(int ic = 0; ic < in_ch; ic++) {
                        // #pragma HLS UNROLL factor=4
                        // compute_kh_kw:
                        for(int kh = 0; kh < k_size; kh++) {
                            // #pragma HLS UNROLL
                            for(int kw = 0; kw < k_size; kw++) {
                                // #pragma HLS UNROLL
                                acc += window_buffer[ic][kh][kw] * weight_buffer[oc][ic][kh][kw];
                            }
                        }
                    }
                    data_t q_val = quantize_acc_to_data(acc);
                    output.write(q_val);
                }
            }
        }
    }
}
