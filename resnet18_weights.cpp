#include "resnet18_weights.h"

data_t block_conv1_weights[BLOCK_CONV1_SIZE] = {
    (data_t)0.0
}; // 512*256*3*3
acc_t block_conv1_bias[BLOCK_CONV1_OUT_C] = {
    (acc_t)0.0
}; // 512
float   block_conv1_bn_mean[OUT_C] = {
    0.0f
}; // 512
float   block_conv1_bn_deno[OUT_C] = {
    0.0f
};
float   block_conv1_bn_gamma[OUT_C] = {
    0.0f
};
float   block_conv1_bn_beta[OUT_C] = {
    0.0f
};

data_t block_conv2_weights[BLOCK_CONV2_SIZE] = {
    (data_t)0.0
}; // 512*512*3*3
acc_t block_conv2_bias[BLOCK_CONV2_OUT_C] = {
    (acc_t)0.0
}; // 512
float   block_conv2_bn_mean[OUT_C] = {
    0.0f
};
float   block_conv2_bn_deno[OUT_C] = {
    0.0f
};
float   block_conv2_bn_gamma[OUT_C] = {
    0.0f
};
float   block_conv2_bn_beta[OUT_C] = {
    0.0f
};

data_t skip_conv_weights[SKIP_CONV_SIZE] = {
    (data_t)0.0
}; // 512*256*3*3
acc_t skip_conv_bias[SKIP_CONV_OUT_C] = {
    (acc_t)0.0
}; // 512
float   skip_bn_mean[OUT_C] = {
    0.0f
};
float   skip_bn_deno[OUT_C] = {
    0.0f
};
float   skip_bn_gamma[OUT_C] = {
    0.0f
};
float   skip_bn_beta[OUT_C] = {
    0.0f
};