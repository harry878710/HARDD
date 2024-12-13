// duplicate_stream.h
#ifndef DUPLICATE_STREAM_H
#define DUPLICATE_STREAM_H

#include "resnet18.h"
#include <hls_stream.h>
#include <ap_fixed.h>

// Define data types
typedef ap_fixed<8,4> data_t;

// Template function to duplicate a stream into two outputs
template <typename T>
void duplicate_stream(
    hls::stream<T> &input_stream,
    hls::stream<T> &out1_stream,
    hls::stream<T> &out2_stream,
    int size
) {
    for(int i = 0; i < size; i++) {
    #pragma HLS PIPELINE II=1
        if(!input_stream.empty()) {
            T data = input_stream.read();
            out1_stream.write(data);
            out2_stream.write(data);
        }
    }
}

#endif // DUPLICATE_STREAM_H
