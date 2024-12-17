// avg_pool.h
#ifndef AVG_POOL_H
#define AVG_POOL_H
#include "resnet18.h"

void avg_pool_1x1(
    hls::stream<data_t> &in_stream,    // Input stream: channels x in_h x in_w
    hls::stream<data_t> &out_stream,   // Output stream: channels x 1 x 1
    int channels,                      // Number of channels
    int in_h,                          // Input in_h
    int in_w                           // Input in_w
);
#endif
