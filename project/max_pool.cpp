// max_pool.cpp
#include "max_pool.h"

// Constants
#define K_SIZE 3
#define STRIDE 2
#define PAD 1
#define IN_H 112
#define IN_W 112
#define IN_CH 64
#define OUT_H 56 // ((IN_H + 2 * PAD - K_SIZE) / STRIDE + 1)
#define OUT_W 56 // ((IN_W + 2 * PAD - K_SIZE) / STRIDE + 1)

// Max Pooling Function: 3x3 kernel, stride=2, pad=1, dilation=1
void max_pool(
    hls::stream<data_t> &in_stream,
    hls::stream<data_t> &out_stream
) {
    // Total padded dimensions
    const int PAD_H = IN_H + 2 * PAD;
    const int PAD_W = IN_W + 2 * PAD;

    // Line buffers to store 3 rows of data
    data_t line_buf0[IN_CH][PAD_W]; // 64*114*3 window to process max pool
    data_t line_buf1[IN_CH][PAD_W];
    data_t line_buf2[IN_CH][PAD_W];
    #pragma HLS ARRAY_PARTITION variable=line_buf0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=line_buf1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=line_buf2 complete dim=1

    // Initialize line buffers
    for (int c = 0; c < IN_CH; c++) {
        for (int w = 0; w < PAD_W; w++) {
            #pragma HLS PIPELINE II=1
            line_buf0[c][w] = 0.0f;
            line_buf1[c][w] = 0.0f;
            line_buf2[c][w] = 0.0f;
        }
    }

    // Process each line of the padded input
    // We'll read line by line. After reading each line, we shift line_buf2->line_buf1->line_buf0.
    for (int h = 0; h < PAD_H; h++) {
        // Shift line buffers down
        for (int w = 0; w < PAD_W; w++) {
            for (int c = 0; c < IN_CH; c++) {
                #pragma HLS PIPELINE II=1
                line_buf2[c][w] = line_buf1[c][w];
                line_buf1[c][w] = line_buf0[c][w];
            }
        }

        // Read the current line into line_buf0 with padding
        for (int w = 0; w < PAD_W; w++) {
            for (int c = 0; c < IN_CH; c++) {
                #pragma HLS PIPELINE II=1
                data_t val = 0.0;
                if ((h >= PAD && h < IN_H + PAD) && (w >= PAD && w < IN_W + PAD)) {
                    val = in_stream.read();
                }
                line_buf0[c][w] = val;
            }
        }

        // Once we have read at least 3 lines, we can start producing output rows
        if (h >= K_SIZE - 1) { // h >= 2 for K_SIZE=3
            int oh = (h - (K_SIZE - 1)) / STRIDE; // output row index
            if (oh < OUT_H && (h - (K_SIZE - 1)) % STRIDE == 0) {
                // Compute output columns
                for (int w = K_SIZE - 1; w < PAD_W; w++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS UNROLL factor=1
                    if ((w - (K_SIZE - 1)) % STRIDE == 0) {
                        int ow = (w - (K_SIZE - 1)) / STRIDE;
                        if (ow < OUT_W) {
                            // Compute the max over the 3x3 window
                            for (int c = 0; c < IN_CH; c++) {
                                data_t max_val = (data_t)-1e9f;
                                #pragma HLS UNROLL
                                for (int kh = 0; kh < K_SIZE; kh++) {
                                    #pragma HLS UNROLL
                                    for (int kw = 0; kw < K_SIZE; kw++) {
                                        data_t val;
                                        if (kh == 0)
                                            val = line_buf2[c][w - (K_SIZE - 1) + kw];
                                        else if (kh == 1)
                                            val = line_buf1[c][w - (K_SIZE - 1) + kw];
                                        else
                                            val = line_buf0[c][w - (K_SIZE - 1) + kw];

                                        if (val > max_val) {
                                            max_val = val;
                                        }
                                    }
                                }
                                // Write the max value for this channel (only 1 CH here)
                                out_stream.write(max_val);
                            }
                        }
                    }
                }
            }
        }
    }
}
