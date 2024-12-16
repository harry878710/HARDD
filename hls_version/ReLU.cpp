// ReLU.cpp
#include "ReLU.h"

void relu(hls::stream<data_t> &in_data, hls::stream<data_t> &out_data, int size) {
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        data_t tmp = in_data.read();
        if (tmp < (data_t)0) {
            out_data.write((data_t)0);
        }
        else{
            out_data.write(tmp);
        }
    }
}
