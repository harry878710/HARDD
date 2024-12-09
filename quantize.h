#ifndef QUANTIZE_H
#define QUANTIZE_H

#include "resnet18.h"
// #include <algorithm>

// Clamp function
inline data_t clamp_to_data_type(acc_t val) {
    if (val < 0) val = 0;
    if (val > 255) val = 255;
    return (data_t)val;
}

#endif
