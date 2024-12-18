// avg_pool.cpp
// Average pooling output_size=(1,1)
#include "avg_pool.h"

void avg_pool_1x1(
    hls::stream<data_t> &in_stream,    // Input stream: channels x in_h x in_w
    hls::stream<data_t> &out_stream,   // Output stream: channels x 1 x 1
    int channels,                      // Number of channels
    int in_h,                          // Input in_h
    int in_w                           // Input in_w
) {
    // Calculate the number of elements per channel
    int in_num = in_h * in_w;

    // Iterate over each channel
    for(int c = 0; c < channels; c++) {
        #pragma HLS PIPELINE II=1
        acc_t sum = 0.0;
        // Calculate sum in the current channel
        for(int ih = 0; ih < in_h; ih++) {
            for(int iw = 0; iw < in_w; iw++) {
                #pragma HLS PIPELINE II=1
                sum += (acc_t)in_stream.read();
            }
        }
        // Compute the average
        acc_t avg = sum / (acc_t)in_num;
        data_t d_avg = clamp_acc_to_data(avg);

        // Write to output stream
        out_stream.write(d_avg);
    }
}
