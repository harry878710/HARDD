// conv.cpp
#include "resnet18.h"

void conv_3x3_stride(
    const data_t *input,
    data_t *output,
    const data_t *weights,
    const acc_t *bias,
    int in_h, int in_w, int in_ch,
    int out_ch, int stride, int pad
) {
    int out_h = in_h / stride;
    int out_w = in_w / stride;

    for (int oc = 0; oc < out_ch; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                acc_t acc = bias[oc]; // bias is in acc type, which is 32-bit
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int ih = oh*stride + kh - pad;
                        int iw = ow*stride + kw - pad;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            for (int ic = 0; ic < in_ch; ic++) {
                                acc_t val = input[ih*in_w*in_ch + iw*in_ch + ic];
                                acc_t w = weights[oc*(in_ch*3*3) + kh*(3*in_ch) + kw*(in_ch) + ic];
                                acc += val * w;
                            }
                        }
                    }
                }
                // output[oc*out_h*out_w + oh*out_w + ow] = (data_t)acc;
                output[oc*out_h*out_w + oh*out_w + ow] = quantize_acc_to_data(acc);
            }
        }
    }
}

void conv_1x1_stride(
    const data_t *input,
    data_t *output,
    const data_t *weights,
    const acc_t *bias,
    int in_h, int in_w, int in_ch,
    int out_ch, int stride
) {
    int out_h = in_h / stride;
    int out_w = in_w / stride;

    for (int oc = 0; oc < out_ch; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                acc_t acc = bias[oc];
                int ih = oh*stride;
                int iw = ow*stride;
                for (int ic = 0; ic < in_ch; ic++) {
                    acc_t val = input[ih*in_w*in_ch + iw*in_ch + ic];
                    acc_t w = weights[oc*(in_ch) + ic];
                    acc += val * w;
                }
                // output[oc*out_h*out_w + oh*out_w + ow] = (data_t)acc;
                output[oc*out_h*out_w + oh*out_w + ow] = quantize_acc_to_data(acc);
            }
        }
    }
}
