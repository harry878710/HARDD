// conv.h
#ifndef CONV_H
#define CONV_H

#include "resnet18.h"
#include "quantize.h"

void conv2D(
    hls::stream<data_t> &input,
    hls::stream<data_t> &output,
    hls::stream<data_t> &weights,
    int in_h, int in_w, int in_ch,
    int out_ch, int k_size, int stride, int pad
);

#endif
