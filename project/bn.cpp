// bn.cpp
#include "bn.h"
#include <cmath>

// Dequantize -> BN -> Requantize
void batch_norm(
    hls::stream<acc_t> &feature_map,
    hls::stream<data_t> &bn_feature_map,
    int H, int W, int C,   // 7x7x512
    hls::stream<float> &mean, hls::stream<float> &deno, hls::stream<float> &gamma, hls::stream<float> &beta
) {
    int size = H*W;
    for (int c = 0; c < C; c++) {
        #pragma HLS pipeline II=1
        float m = mean.read();
        float d = deno.read();
        float g = gamma.read();
        float b = beta.read();
        for (int i = 0; i < size; i++) {
            #pragma HLS pipeline II=1
            // int idx = c*size + i;
            // Dequantize
            float val = dequantize_acc_to_float(feature_map.read());
            
            // BN
            // val = ((val - m) / d) * g + b;  // d = sqrtf(var + epsilon))
            float tmp1 = val - m;
            float tmp2 = tmp1 / d;
            float tmp3 = tmp2 * g;
            float tmp4 = tmp3 + b;
            // val = tmp4;

            // Quantize back to acc_t
            acc_t qval = quantize_float_to_acc(tmp4);
            data_t d_val = clamp_acc_to_data(q_val);
            bn_feature_map.write(d_val);
        }
    }
}
