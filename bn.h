// bn.h / bn.cpp
#ifndef BN_H
#define BN_H

#include "resnet18.h"
#include "quantize.h"

void batch_norm(
    data_t *feature_map,
    int H, int W, int C,
    const float *mean, const float *deno, const float *gamma, const float *beta
);

#endif
