// conv.h
#ifndef CONV_H
#define CONV_H

#include "resnet18.h"

void conv_3x3_stride(
    const data_t *input,
    data_t *output,
    const data_t *weights,
    const acc_t *bias, // biases often stored as int32 for quantized models
    int in_h, int in_w, int in_ch,
    int out_ch, int stride, int pad
);

void conv_1x1_stride(
    const data_t *input,
    data_t *output,
    const data_t *weights,
    const acc_t *bias,
    int in_h, int in_w, int in_ch,
    int out_ch, int stride
);

#endif
