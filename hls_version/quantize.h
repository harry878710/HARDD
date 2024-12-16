#ifndef QUANTIZE_H
#define QUANTIZE_H

#include "resnet18.h"
// #include <algorithm>

// Clamp function
inline float dequantize_data_to_float(data_t val){
    float fval = (float)val;
    return fval;
}

inline float dequantize_data_to_acc(data_t val){
    acc_t fval = (acc_t)val;
    return fval;
}

inline data_t quantize_acc_to_data(acc_t val){
    // Multiply by scale
    acc_t scaled = val * (acc_t)IN_SCALE;

    // Round to nearest integer
    float f_scaled = (float)scaled;
    acc_t q_val = (acc_t)std::round(f_scaled);
    // acc_t q_val = scaled;
    
    // Clip to the representable range
    // scaled data_t can represent [-128, 127]
    if (q_val < -128) q_val = -128;
    if (q_val > 127)  q_val = 127;
    
    // Convert to data_t
    acc_t qs_val = q_val / acc_t(OUT_SCALE);
    return (data_t)qs_val;
}

inline data_t quantize_float_to_data(float val){
    // Multiply by scale
    float scaled = val * IN_SCALE;

    // Round to nearest integer
    float f_scaled = (float)scaled;
    float q_val = (float)std::round(f_scaled);
    // float q_val = scaled;
    
    // Clamp to the representable range
    // ap_fixed<8,4> can represent [-8.0, 7]
    // Integer range after scaling by 16 is [-128, 127]
    if (q_val < -128) q_val = -128;
    if (q_val > 127)  q_val = 127;

    float qs_val = q_val/OUT_SCALE;
    // Convert to data_t
    return (data_t)qs_val;
}

#endif
