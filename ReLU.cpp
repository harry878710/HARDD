// ReLU.cpp
#include "ReLU.h"

void relu(data_t *data, int size) {
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        if (data[i] < (data_t)0) {
            data[i] = (data_t)0;
        }
    }
}
