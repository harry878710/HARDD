#ifndef QUANTIZE_H
#define QUANTIZE_H

#include "resnet18.h"
// #include <algorithm>

// Clamp function
inline float dequantize_data_to_float(data_t val){
    float fval = (float)val;
    return fval;
}

inline data_t quantize_acc_to_data(acc_t val){
    // Multiply by scale
    acc_t scaled = val * (acc_t)IN_SCALE;
    
    // Clip to the representable range
    // ap_fixed<16,8> can represent [-128, 127]
    // Integer range after scaling by 16 is [-2048, 2047]
    if (scaled < -2048) scaled = -2048;
    if (scaled > 2047)  scaled = 2047;
    
    // Convert to data_t
    data_t qval = (data_t)(scaled / acc_t(OUT_SCALE));
    return qval;
}

inline data_t quantize_float_to_data(float val){
    // Multiply by scale
    float scaled = val * IN_SCALE;
    
    // Clip to the representable range
    // ap_fixed<8,4> can represent [-8.0, 7]
    // Integer range after scaling by 16 is [-128, 127]
    if (scaled < -128) scaled = -128;
    if (scaled > 127)  scaled = 127;
    
    // Convert to data_t
    data_t qval = (data_t)(scaled / OUT_SCALE);
    return qval;
}

#endif
