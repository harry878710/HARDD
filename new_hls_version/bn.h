// bn.h / bn.cpp
#ifndef BN_H
#define BN_H

#include "resnet18.h"
#include "quantize.h"

void batch_norm(
    hls::stream<data_t> &feature_map,
    hls::stream<data_t> &bn_feature_map,
    int H, int W, int C,   // 7x7x512
    hls::stream<float> &mean, hls::stream<float> &deno, hls::stream<float> &gamma, hls::stream<float> &beta
);

#endif
