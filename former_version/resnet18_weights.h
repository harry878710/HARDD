#ifndef RESNET18_WEIGHTS_H
#define RESNET18_WEIGHTS_H
#include "resnet18.h"

template<typename T>
bool load_array_from_file(hls::stream<T> &filename, const T* arr, int size);

#endif
