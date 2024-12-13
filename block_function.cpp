// block_function.cpp
#include "block_function.h"
#include "resnet18_weights.h"

// External declarations of weights and BN parameters.
// These should be defined in resnet18_weights.h as quantized int8_t for weights, int32_t for bias, and float for BN.
extern data_t block_conv1_weights[];
// extern acc_t block_conv1_bias[];
extern float   block_conv1_bn_mean[];
extern float   block_conv1_bn_deno[];
extern float   block_conv1_bn_gamma[];
extern float   block_conv1_bn_beta[];

extern data_t block_conv2_weights[];
// extern acc_t block_conv2_bias[];
extern float   block_conv2_bn_mean[];
extern float   block_conv2_bn_deno[];
extern float   block_conv2_bn_gamma[];
extern float   block_conv2_bn_beta[];

extern data_t skip_conv_weights[];
// extern acc_t skip_conv_bias[];
extern float   skip_bn_mean[];
extern float   skip_bn_deno[];
extern float   skip_bn_gamma[];
extern float   skip_bn_beta[];

void run_resnet_block(
    const data_t* input,
    data_t* output,
    int in_h, int in_w, int in_ch, int out_ch  // H*W=14x14 in_ch=256 out_ch=512
) {
    int out_h = in_h / 2; // stride=2 in first conv
    int out_w = in_w / 2; // out_h = out_w = 7
    int out_size = out_h*out_w*out_ch; // 7*7*512

    static data_t out1[IN_H*IN_W*OUT_C]; // Adjust sizes as needed, here is max
    static data_t out2[IN_H*IN_W*OUT_C];
    static data_t skip_out[IN_H*IN_W*OUT_C];

    // MAIN PATH
    // Conv 3x3 stride 2
    // conv_3x3_stride(input, out1, block_conv1_weights, block_conv1_bias, in_h, in_w, in_ch, out_ch, 2, 1);
    conv_3x3_stride(input, out1, block_conv1_weights, in_h, in_w, in_ch, out_ch, 2, 1);
    // for (int i = 0; i < 10; i++) {
    //     printf("Out 1[%d]: %f\n", i, (float)out1[i]);
    // }
    // BN + ReLU
    batch_norm(out1, out_h, out_w, out_ch, block_conv1_bn_mean, block_conv1_bn_deno, block_conv1_bn_gamma, block_conv1_bn_beta);
    relu(out1, out_h*out_w*out_ch);
    // Conv 3x3 stride 1
    // conv_3x3_stride(out1, out2, block_conv2_weights, block_conv2_bias, out_h, out_w, out_ch, out_ch, 1, 1);
    conv_3x3_stride(out1, out2, block_conv2_weights, out_h, out_w, out_ch, out_ch, 1, 1);
    // BN (no ReLU yet)
    batch_norm(out2, out_h, out_w, out_ch, block_conv2_bn_mean, block_conv2_bn_deno, block_conv2_bn_gamma, block_conv2_bn_beta);

    // SKIP PATH:
    // conv_1x1_stride(input, skip_out, skip_conv_weights, skip_conv_bias, in_h, in_w, in_ch, out_ch, 2);
    conv_1x1_stride(input, skip_out, skip_conv_weights, in_h, in_w, in_ch, out_ch, 2);
    batch_norm(skip_out, out_h, out_w, out_ch, skip_bn_mean, skip_bn_deno, skip_bn_gamma, skip_bn_beta);

    // Add main_path + skip_path
    skip_add(out2, skip_out, out_size);

    // Final ReLU
    relu(out2, out_size);

    // out2 is the final output
    for (int i = 0; i < out_size; i++) {
        output[i] = out2[i];
    }
}
