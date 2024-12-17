// max_pool.cpp
#include "max_pool.h"

// Max Pooling Function: 3x3 kernel, stride 2, padding 1, dilation 1
void max_pool(
    hls::stream<data_t> &in_stream,
    hls::stream<data_t> &out_stream,
    int in_h,
    int in_w,
    int in_ch
) {
    // Calculate output dimensions
    int out_h = std::floor((in_h + 2 * 1 - 3) / 2) + 1;  // (in_h + 2*pad-kernel)/stride +1
    int out_w  = std::floor((in_w + 2 * 1 - 3) / 2) + 1; // (112+2*1-3)/2 + 1 = 56

    // Temporary buffer to store a 3-row window
    // Including padding on both sides
    data_t line_buffer[3][(in_w + 2)];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1

    // Initialize line buffer (3 row)
    for(int c = 0; c < in_ch; c++) {
        // upper padding
        for(int i = 0; i < in_w + 2; i++) {
            line_buffer[0][i] = 0.0;
        }
        // Read first row with padding
        for(int i = 0; i < in_w + 2; i++) {
            if(i == 0 || i == in_w +1) {
                line_buffer[1][i] = 0.0; // Left and right padding
            }
            else {
                line_buffer[1][i] = in_stream.read();
            }
        }
        // Read second row with padding
        for(int i = 0; i < in_w + 2; i++) {
            if(i == 0 || i == in_w +1) {
                line_buffer[2][i] = 0.0; // Left and right padding
            }
            else {
                line_buffer[2][i] = in_stream.read();
            }
        }

        // Process each subsequent row
        for(int h = 0; h < in_h; h++) {
            // Read the next row with padding
            for(int i = 0; i < in_w + 2; i++) {
                if(i == 0 || i == in_w +1) {
                    line_buffer[2][i] = 0.0; // Left and right padding
                }
                else {
                    line_buffer[2][i] = in_stream.read();
                }
            }

            // Only perform pooling on even-indexed rows (stride=2)
            if(h % 2 == 0) {
                for(int w = 0; w < out_w; w++) {
                    #pragma HLS PIPELINE II=1
                    // Extract the 3x3 window
                    data_t window[3][3];
                    for(int m = 0; m < 3; m++) {
                        for(int n = 0; n < 3; n++) {
                            window[m][n] = line_buffer[m][w + n];
                        }
                    }

                    // Find the maximum value in the 3x3 window
                    data_t max_val = window[0][0];
                    for(int m = 0; m < 3; m++) {
                        for(int n = 0; n < 3; n++) {
                            if(window[m][n] > max_val) {
                                max_val = window[m][n];
                            }
                        }
                    }

                    // Write the maximum value to the output stream
                    out_stream.write(max_val);
                }
            }

            // Shift the line buffer up for the next window
            for(int m = 0; m < 2; m++) {
                for(int i = 0; i < in_w + 2; i++) {
                    #pragma HLS UNROLL
                    line_buffer[m][i] = line_buffer[m + 1][i];
                }
            }
        }
    }
}


