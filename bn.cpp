// bn.cpp
#include "bn.h"
#include <cmath>

// Dequantize -> BN -> Requantize
void batch_norm(
    data_t *feature_map,
    int H, int W, int C,   // 7x7x512
    const float *mean, const float *deno, const float *gamma, const float *beta
) {
    int size = H*W;
    for (int c = 0; c < C; c++) {
        float m = mean[c];
        float d = deno[c];
        float g = gamma[c];
        float b = beta[c];
        for (int i = 0; i < size; i++) {
            int idx = c*size + i;
            // Dequantize
            float val = (float)feature_map[idx];
            
            // BN
            // val = ((val - m) / d) * g + b;  // d = sqrtf(var + epsilon))
            float tmp1 = val - m;
            float tmp2 = tmp1 / d;
            float tmp3 = tmp2 * g;
            float tmp4 = tmp3 + b;
            val = tmp4;

            // Requantize
            // acc_t qval = (acc_t)val;
            feature_map[idx] = (data_t)val;
        }
    }
}
