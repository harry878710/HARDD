// block_function.h
#ifndef BLOCK_FUNCTION_H
#define BLOCK_FUNCTION_H

#include "resnet18.h"
#include "resnet18_weights.h" // Quantized weights and BN parameters
#include "conv.h"
#include "bn.h"
#include "ReLU.h"
#include "skipConnect.h"
#include "quantize.h"
#include "duplicate_stream.h"
#include <ap_fixed.h>
#include <hls_stream.h>

void run_resnet_block(
    hls::stream<data_t> &input,
    hls::stream<data_t> &output,
    hls::stream<data_t> &conv_1_weights_stream,
    hls::stream<data_t> &conv_2_weights_stream,
    hls::stream<data_t> &conv_s_weights_stream,
    hls::stream<float>  &block_conv1_bn_mean_stream,
    hls::stream<float>  &block_conv1_bn_deno_stream,
    hls::stream<float>  &block_conv1_bn_gamma_stream,
    hls::stream<float>  &block_conv1_bn_beta_stream,
    hls::stream<float>  &block_conv2_bn_mean_stream,
    hls::stream<float>  &block_conv2_bn_deno_stream,
    hls::stream<float>  &block_conv2_bn_gamma_stream,
    hls::stream<float>  &block_conv2_bn_beta_stream,
    hls::stream<float>  &skip_bn_mean_stream,
    hls::stream<float>  &skip_bn_deno_stream,
    hls::stream<float>  &skip_bn_gamma_stream,
    hls::stream<float>  &skip_bn_beta_stream,
    int in_h, int in_w, int in_ch, int out_ch  // H*W=14x14 in_ch=256 out_ch=512
);

#endif
