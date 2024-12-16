#ifndef RESNET18_WEIGHTS_H
#define RESNET18_WEIGHTS_H
#include "resnet18.h"
#include <ap_fixed.h> 

template<typename T>
bool load_array_to_stream(hls::stream<T> &data_stream, const T* arr, int size) {
    if (arr == nullptr || size <= 0) {
        return false;
    }
    for (int i = 0; i < size; i++) {
       data_stream.write(arr[i]);
    }
    return true;
}
#endif
