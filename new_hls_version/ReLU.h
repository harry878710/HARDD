// ReLU.h
#ifndef RELU_H
#define RELU_H

#include "resnet18.h"

void relu(hls::stream<data_t> &in_data, hls::stream<data_t> &out_data, int size);

#endif
