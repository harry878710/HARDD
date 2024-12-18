#include "max_pool.h"
// #include <hls_stream.h>

// Constants
#define K_SIZE 3
#define STRIDE 2
#define PAD 1
#define DILATION 1
#define IN_H 4
#define IN_W 4
#define IN_CH 1
#define OUT_H ((IN_H + 2 * PAD - K_SIZE) / STRIDE + 1)
#define OUT_W ((IN_W + 2 * PAD - K_SIZE) / STRIDE + 1)

// Max Pooling Function: 3x3 kernel, stride 2, padding 1, dilation 1
void max_pool(
    hls::stream<float> &in_stream,
    hls::stream<float> &out_stream
) {
    // Create a buffer to hold the padded input
    float input_buf[IN_CH][IN_H + 2 * PAD][IN_W + 2 * PAD];
    #pragma HLS ARRAY_PARTITION variable=input_buf complete dim=1

    // Read the input into the buffer with zero-padding
    for (int h = 0; h < IN_H + 2*PAD; h++) {
        for (int w = 0; w < IN_W + 2*PAD; w++) {
            for (int c = 0; c < IN_CH; c++) {
                #pragma HLS PIPELINE II=1
                float val = 0.0f;
                // If within the input region (considering the padding)
                if (h >= PAD && h < IN_H + PAD && w >= PAD && w < IN_W + PAD) {
                    val = in_stream.read();
                }
                input_buf[c][h][w] = val;
            }
        }
    }

    // Perform max pooling
    for (int oh = 0; oh < OUT_H; oh++) {
        for (int ow = 0; ow < OUT_W; ow++) {
            for (int c = 0; c < IN_CH; c++) {
                float max_val = -1e9;
                // Iterate over the 3x3 kernel
                for (int kh = 0; kh < K_SIZE; kh++) {
                    for (int kw = 0; kw < K_SIZE; kw++) {
                        #pragma HLS UNROLL
                        int ih = oh * STRIDE + kh;
                        int iw = ow * STRIDE + kw;
                        float val = input_buf[c][ih][iw];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                out_stream.write(max_val);
            }
        }
    }
}