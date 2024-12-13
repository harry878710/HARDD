// resnet18_weights.cpp
#include "resnet18_weights.h"

static bool arrays_initialized = false;

template<typename T>
bool load_array_from_file(hls::stream<T> &filename, const T* arr, int size) {
    if (arr == nullptr || size <= 0) {
        return false;
    }
    for (int i = 0; i < size; i++) {
       filename.write(arr[i]);
    }
    return true;
}

