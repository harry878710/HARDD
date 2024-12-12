#ifndef RESNET18_WEIGHTS_H
#define RESNET18_WEIGHTS_H
#include "resnet18.h"

extern data_t block_conv1_weights[BLOCK_CONV1_SIZE]; // 512*256*3*3
extern acc_t block_conv1_bias[BLOCK_CONV1_OUT_C]; // 512
extern float   block_conv1_bn_mean[OUT_C]; // 512
extern float   block_conv1_bn_deno[OUT_C];
extern float   block_conv1_bn_gamma[OUT_C];
extern float   block_conv1_bn_beta[OUT_C];

extern data_t block_conv2_weights[BLOCK_CONV2_SIZE]; // 512*512*3*3
extern acc_t block_conv2_bias[BLOCK_CONV2_OUT_C]; // 512
extern float   block_conv2_bn_mean[OUT_C];
extern float   block_conv2_bn_deno[OUT_C];
extern float   block_conv2_bn_gamma[OUT_C];
extern float   block_conv2_bn_beta[OUT_C];

extern data_t skip_conv_weights[SKIP_CONV_SIZE]; // 512*256*3*3
extern acc_t skip_conv_bias[SKIP_CONV_OUT_C]; // 512
extern float   skip_bn_mean[OUT_C];
extern float   skip_bn_deno[OUT_C];
extern float   skip_bn_gamma[OUT_C];
extern float   skip_bn_beta[OUT_C];

template<typename T>
bool load_array_from_file(const std::string &filename, T* arr, int size);
bool initialize_arrays();

#endif
