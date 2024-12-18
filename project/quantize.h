#ifndef QUANTIZE_H
#define QUANTIZE_H

#include "resnet18.h"

// Clamp function

// inline float dequantize_data_to_acc(data_t val){
//     acc_t fval = (acc_t)val;
//     return fval;
// }

inline data_t clamp_acc_to_data(acc_t val){
    if(val>0 && val > MAX_DATA_T){
        return MAX_DATA_T; // Based on ap_fixed precision
    }
    else if(val<0 && val < MIN_DATA_T){
        return MIN_DATA_T; // Based on ap_fixed precision
    }
    else {
        return (data_t)val;
    }
}




// Quantize from float to data_t
inline data_t quantize_float_to_data(float val) {
    return (data_t)(val * DATA_SCALE);
}

// Dequantize data_t to float
inline float dequantize_data_to_float(data_t q) {
    return ((float)q) / DATA_SCALE;
}

// Quantize float to acc_t
inline acc_t quantize_float_to_acc(float val) {
    return (acc_t)(val * ACC_SCALE);
}

// Dequantize acc_t to float
inline float dequantize_acc_to_float(acc_t q) {
    return ((float)q) / ACC_SCALE;
}

#endif
