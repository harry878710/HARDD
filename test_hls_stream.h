// test_hls_stream.h
#ifndef TEST_HLS_MODULE_H
#define TEST_HLS_MODULE_H

#include <hls_stream.h>

template <typename T>
void test_hls_module(hls::stream<T> &input_stream, hls::stream<T> &output_stream, int size) {
    for (int i = 0; i < size; ++i) {
#pragma HLS PIPELINE
        if (!input_stream.empty()) {
            T data = input_stream.read();
            output_stream.write(data);
        }
    }
}

#endif // TEST_HLS_MODULE_H
