// max_pool.h
#ifndef MAX_POOL_H
#define MAX_POOL_H

#include "resnet18.h"

void max_pool(
    hls::stream<data_t> &in_stream,
    hls::stream<data_t> &out_stream,
    int in_h,
    int in_w,
    int in_ch
);
#endif