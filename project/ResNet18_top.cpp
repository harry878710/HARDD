// ResNet18_top.cpp
#include <hls_stream.h>
#include <ap_fixed.h>
#include <iostream>
#include "resnet18_weights.h"
#include "ResNet18_top.h"

void run_ResNet_top(
    hls::stream<data_t> &input,
    hls::stream<data_t> &output,
    hls::stream<data_t> &layer0_conv_1_weights_stream,
    hls::stream<data_t> &layer1_conv_1_weights_stream,
    hls::stream<data_t> &layer1_conv_2_weights_stream,
    hls::stream<data_t> &layer1_conv_3_weights_stream,
    hls::stream<data_t> &layer2_conv_1_weights_stream,
    hls::stream<data_t> &layer2_conv_2_weights_stream,
    hls::stream<data_t> &layer2_conv_3_weights_stream,
    hls::stream<data_t> &layer2_conv_4_weights_stream,
    hls::stream<data_t> &layer3_conv_1_weights_stream,
    hls::stream<data_t> &layer3_conv_2_weights_stream,
    hls::stream<data_t> &layer3_conv_3_weights_stream,
    hls::stream<data_t> &layer3_conv_4_weights_stream,
    hls::stream<data_t> &layer4_conv_1_weights_stream,
    hls::stream<data_t> &layer2_conv_s_weights_stream,
    hls::stream<data_t> &layer3_conv_s_weights_stream,
    hls::stream<data_t> &conv_2_weights_stream,
    hls::stream<data_t> &conv_s_weights_stream,
    hls::stream<float>  &layer0_conv1_bn_mean_stream,
    hls::stream<float>  &layer0_conv1_bn_deno_stream,
    hls::stream<float>  &layer0_conv1_bn_gamma_stream,
    hls::stream<float>  &layer0_conv1_bn_beta_stream,
    hls::stream<float>  &layer1_conv1_bn_mean_stream,
    hls::stream<float>  &layer1_conv1_bn_deno_stream,
    hls::stream<float>  &layer1_conv1_bn_gamma_stream,
    hls::stream<float>  &layer1_conv1_bn_beta_stream,
    hls::stream<float>  &layer1_conv2_bn_mean_stream,
    hls::stream<float>  &layer1_conv2_bn_deno_stream,
    hls::stream<float>  &layer1_conv2_bn_gamma_stream,
    hls::stream<float>  &layer1_conv2_bn_beta_stream,
    hls::stream<float>  &layer1_conv3_bn_mean_stream,
    hls::stream<float>  &layer1_conv3_bn_deno_stream,
    hls::stream<float>  &layer1_conv3_bn_gamma_stream,
    hls::stream<float>  &layer1_conv3_bn_beta_stream,
    hls::stream<float>  &layer1_conv4_bn_mean_stream,
    hls::stream<float>  &layer1_conv4_bn_deno_stream,
    hls::stream<float>  &layer1_conv4_bn_gamma_stream,
    hls::stream<float>  &layer1_conv4_bn_beta_stream,
    hls::stream<float>  &layer2_conv1_bn_mean_stream,
    hls::stream<float>  &layer2_conv1_bn_deno_stream,
    hls::stream<float>  &layer2_conv1_bn_gamma_stream,
    hls::stream<float>  &layer2_conv1_bn_beta_stream,
    hls::stream<float>  &layer2_conv2_bn_mean_stream,
    hls::stream<float>  &layer2_conv2_bn_deno_stream,
    hls::stream<float>  &layer2_conv2_bn_gamma_stream,
    hls::stream<float>  &layer2_conv2_bn_beta_stream,
    hls::stream<float>  &layer2_conv3_bn_mean_stream,
    hls::stream<float>  &layer2_conv3_bn_deno_stream,
    hls::stream<float>  &layer2_conv3_bn_gamma_stream,
    hls::stream<float>  &layer2_conv3_bn_beta_stream,
    hls::stream<float>  &layer2_conv4_bn_mean_stream,
    hls::stream<float>  &layer2_conv4_bn_deno_stream,
    hls::stream<float>  &layer2_conv4_bn_gamma_stream,
    hls::stream<float>  &layer2_conv4_bn_beta_stream,
    hls::stream<float>  &layer3_conv1_bn_mean_stream,
    hls::stream<float>  &layer3_conv1_bn_deno_stream,
    hls::stream<float>  &layer3_conv1_bn_gamma_stream,
    hls::stream<float>  &layer3_conv1_bn_beta_stream,
    hls::stream<float>  &layer3_conv2_bn_mean_stream,
    hls::stream<float>  &layer3_conv2_bn_deno_stream,
    hls::stream<float>  &layer3_conv2_bn_gamma_stream,
    hls::stream<float>  &layer3_conv2_bn_beta_stream,
    hls::stream<float>  &layer3_conv3_bn_mean_stream,
    hls::stream<float>  &layer3_conv3_bn_deno_stream,
    hls::stream<float>  &layer3_conv3_bn_gamma_stream,
    hls::stream<float>  &layer3_conv3_bn_beta_stream,
    hls::stream<float>  &layer3_conv4_bn_mean_stream,
    hls::stream<float>  &layer3_conv4_bn_deno_stream,
    hls::stream<float>  &layer3_conv4_bn_gamma_stream,
    hls::stream<float>  &layer3_conv4_bn_beta_stream,
    hls::stream<float>  &layer4_conv1_bn_mean_stream,
    hls::stream<float>  &layer4_conv1_bn_deno_stream,
    hls::stream<float>  &layer4_conv1_bn_gamma_stream,
    hls::stream<float>  &layer4_conv1_bn_beta_stream,
    hls::stream<float>  &layer4_conv2_bn_mean_stream,
    hls::stream<float>  &layer4_conv2_bn_deno_stream,
    hls::stream<float>  &layer4_conv2_bn_gamma_stream,
    hls::stream<float>  &layer4_conv2_bn_beta_stream,
    hls::stream<float>  &layer4_conv3_bn_mean_stream,
    hls::stream<float>  &layer4_conv3_bn_deno_stream,
    hls::stream<float>  &layer4_conv3_bn_gamma_stream,
    hls::stream<float>  &layer4_conv3_bn_beta_stream,
    hls::stream<float>  &layer4_conv4_bn_mean_stream,
    hls::stream<float>  &layer4_conv4_bn_deno_stream,
    hls::stream<float>  &layer4_conv4_bn_gamma_stream,
    hls::stream<float>  &layer4_conv4_bn_beta_stream,
    hls::stream<float>  &block_conv1_bn_beta_stream,
    hls::stream<float>  &block_conv2_bn_mean_stream,
    hls::stream<float>  &block_conv2_bn_deno_stream,
    hls::stream<float>  &block_conv2_bn_gamma_stream,
    hls::stream<float>  &block_conv2_bn_beta_stream,
    hls::stream<float>  &skip_layer2_bn_mean_stream,
    hls::stream<float>  &skip_layer2_bn_deno_stream,
    hls::stream<float>  &skip_layer2_bn_gamma_stream,
    hls::stream<float>  &skip_layer2_bn_beta_stream,
    hls::stream<float>  &skip_layer3_bn_mean_stream,
    hls::stream<float>  &skip_layer3_bn_deno_stream,
    hls::stream<float>  &skip_layer3_bn_gamma_stream,
    hls::stream<float>  &skip_layer3_bn_beta_stream,
    hls::stream<float>  &skip_layer4_bn_mean_stream,
    hls::stream<float>  &skip_layer4_bn_deno_stream,
    hls::stream<float>  &skip_layer4_bn_gamma_stream,
    hls::stream<float>  &skip_layer4_bn_beta_stream,
    hls::stream<data_t> fc_weights_stream,
    hls::stream<data_t> fc_bias_stream

) {

// Define local streams as objects, not references

//Layer0

    hls::stream<data_t> layer0_input("layer0_input");
    #pragma HLS STREAM variable=layer0_input depth=16

    hls::stream<acc_t> layer0_7x7_stride2("layer0_7x7_stride2");
    #pragma HLS STREAM variable=layer0_7x7_stride2 depth=16

    hls::stream<data_t> layer0_7x7_stride2_bn("layer0_7x7_stride2_bn");
    #pragma HLS STREAM variable=layer0_7x7_stride2_bn depth=16

    hls::stream<data_t> layer0_7x7_stride2_relu("layer0_7x7_stride2_relu");
    #pragma HLS STREAM variable=layer0_7x7_stride2_relu depth=16
    
    hls::stream<data_t> layer0_7x7_stride2_mp("layer0_7x7_stride2_mp");
    #pragma HLS STREAM variable=layer0_7x7_stride2_mp depth=16

//Layer1

    hls::stream<acc_t> layer1_3x3_stride1("layer1_3x3_stride1");
    #pragma HLS STREAM variable=layer1_3x3_stride1 depth=16

    hls::stream<data_t> layer1_3x3_stride1_bn("layer1_3x3_stride1_bn");
    #pragma HLS STREAM variable=layer1_3x3_stride1_bn depth=16

    hls::stream<data_t> layer1_3x3_stride1_relu("layer1_3x3_stride1_relu");
    #pragma HLS STREAM variable=layer1_3x3_stride1_relu depth=16
    
    hls::stream<acc_t> layer1_3x3_stride1_conv2("layer1_3x3_stride1_conv2");
    #pragma HLS STREAM variable=layer1_3x3_stride1_conv2 depth=16

    hls::stream<data_t> layer1_3x3_stride1_conv2_bn("layer1_3x3_stride1_conv2_bn");
    #pragma HLS STREAM variable=layer1_3x3_stride1_conv2_bn depth=16

    hls::stream<acc_t> layer1_3x3_stride1_conv3("layer1_3x3_stride1_conv3");
    #pragma HLS STREAM variable=layer1_3x3_stride1_conv3 depth=16

    hls::stream<data_t> layer1_3x3_stride1_conv3_bn("layer1_3x3_stride1_conv3_bn");
    #pragma HLS STREAM variable=layer1_3x3_stride1_conv3_bn depth=16

    hls::stream<data_t> layer1_3x3_stride1_relu_conv3("layer1_3x3_stride1_relu_conv3");
    #pragma HLS STREAM variable=layer1_3x3_stride1_relu_conv3 depth=16

    hls::stream<acc_t> layer1_3x3_stride1_conv4("layer1_3x3_stride1_conv4");
    #pragma HLS STREAM variable=layer1_3x3_stride1_conv4 depth=16

    hls::stream<data_t> layer1_3x3_stride1_conv4_bn("layer1_3x3_stride1_conv4_bn");
    #pragma HLS STREAM variable=layer1_3x3_stride1_conv4_bn depth=16

// Layer 2

    hls::stream<acc_t> layer2_3x3_stride2("layer2_3x3_stride2");
    #pragma HLS STREAM variable=layer2_3x3_stride2 depth=16

    hls::stream<data_t> layer2_3x3_stride2_bn("layer2_3x3_stride2_bn");
    #pragma HLS STREAM variable=layer2_3x3_stride2_bn depth=16

    hls::stream<data_t> layer2_3x3_stride2_relu("layer2_3x3_stride2_relu");
    #pragma HLS STREAM variable=layer2_3x3_stride2_relu depth=16

    hls::stream<acc_t> layer2_3x3_stride2_conv2("layer2_3x3_stride2_conv2");
    #pragma HLS STREAM variable=layer2_3x3_stride2_conv2 depth=16

    hls::stream<data_t> layer2_3x3_stride2_conv2_bn("layer2_3x3_stride2_conv2_bn");
    #pragma HLS STREAM variable=layer2_3x3_stride2_conv2_bn depth=16

    hls::stream<acc_t> skip_layer2_3x3_stride2("skip_layer2_3x3_stride2");
    #pragma HLS STREAM variable=skip_layer2_3x3_stride2 depth=16

    hls::stream<data_t> skip_layer2_3x3_stride2_bn("skip_layer2_3x3_stride2_bn");
    #pragma HLS STREAM variable=skip_layer2_3x3_stride2_bn depth=16

    hls::stream<data_t> layer2_connected_path("layer2_connected_path");
    #pragma HLS STREAM variable=layer2_connected_path depth=16

    hls::stream<acc_t> layer2_3x3_stride2_conv3("layer2_3x3_stride2_conv3");
    #pragma HLS STREAM variable=layer2_3x3_stride2_conv3 depth=16

    hls::stream<data_t> layer2_3x3_stride2_conv3_bn("layer2_3x3_stride2_conv3_bn");
    #pragma HLS STREAM variable=layer2_3x3_stride2_conv3_bn depth=16

    hls::stream<data_t> layer2_3x3_stride2_relu_conv3("layer2_3x3_stride2_relu_conv3");
    #pragma HLS STREAM variable=layer2_3x3_stride2_relu_conv3 depth=16

    hls::stream<acc_t> layer2_3x3_stride2_conv4("layer2_3x3_stride2_conv4");
    #pragma HLS STREAM variable=layer2_3x3_stride2_conv4 depth=16

    hls::stream<data_t> layer2_3x3_stride2_conv4_bn("layer2_3x3_stride2_conv4_bn");
    #pragma HLS STREAM variable=layer2_3x3_stride2_conv4_bn depth=16
    
    hls::stream<data_t> layer2_3x3_stride2_conv4_bn1("layer2_3x3_stride2_conv4_bn1");
    #pragma HLS STREAM variable=layer2_3x3_stride2_conv4_bn1 depth=16
    
    hls::stream<data_t> layer2_3x3_stride2_conv4_bn2("layer2_3x3_stride2_conv4_bn2");
    #pragma HLS STREAM variable=layer2_3x3_stride2_conv4_bn2 depth=16

// Layer 3

    hls::stream<acc_t> layer3_3x3_stride2("layer3_3x3_stride2");
    #pragma HLS STREAM variable=layer3_3x3_stride2 depth=16
    
    hls::stream<data_t> layer3_3x3_stride2_bn("layer3_3x3_stride2_bn");
    #pragma HLS STREAM variable=layer3_3x3_stride2_bn depth=16
    
    hls::stream<data_t> layer3_3x3_stride2_relu("layer3_3x3_stride2_relu");
    #pragma HLS STREAM variable=layer3_3x3_stride2_relu depth=16

    hls::stream<acc_t> layer3_3x3_stride2_conv2("layer3_3x3_stride2_conv2");
    #pragma HLS STREAM variable=layer3_3x3_stride2_conv2 depth=16
    
    hls::stream<data_t> layer3_3x3_stride2_conv2_bn("layer3_3x3_stride2_conv2_bn");
    #pragma HLS STREAM variable=layer3_3x3_stride2_conv2_bn depth=16

    hls::stream<acc_t> skip_layer3_3x3_stride2("skip_layer3_3x3_stride2");
    #pragma HLS STREAM variable=skip_layer3_3x3_stride2 depth=16

    hls::stream<data_t> skip_layer3_3x3_stride2_bn("skip_layer3_3x3_stride2_bn");
    #pragma HLS STREAM variable=skip_layer3_3x3_stride2_bn depth=16
    
    hls::stream<data_t> layer3_connected_path("layer3_connected_path");
    #pragma HLS STREAM variable=layer3_connected_path depth=16
    
    hls::stream<acc_t> layer3_3x3_stride2_conv3("layer3_3x3_stride2_conv3");
    #pragma HLS STREAM variable=layer3_3x3_stride2_conv3 depth=16
    
    hls::stream<data_t> layer3_3x3_stride2_conv3_bn("layer3_3x3_stride2_conv3_bn");
    #pragma HLS STREAM variable=layer3_3x3_stride2_conv3_bn depth=16
    
    hls::stream<data_t> layer3_3x3_stride2_relu_conv3("layer3_3x3_stride2_relu_conv3");
    #pragma HLS STREAM variable=layer3_3x3_stride2_relu_conv3 depth=16
    
    hls::stream<acc_t> layer3_3x3_stride2_conv4("layer3_3x3_stride2_conv4");
    #pragma HLS STREAM variable=layer3_3x3_stride2_conv4 depth=16
    
    hls::stream<data_t> layer3_3x3_stride2_conv4_bn("layer3_3x3_stride2_conv4_bn");
    #pragma HLS STREAM variable=layer3_3x3_stride2_conv4_bn depth=16
    
    hls::stream<data_t> layer3_3x3_stride2_conv4_bn1("layer3_3x3_stride2_conv4_bn1");
    #pragma HLS STREAM variable=layer3_3x3_stride2_conv4_bn1 depth=16

    hls::stream<data_t> layer3_3x3_stride2_conv4_bn2("layer3_3x3_stride2_conv4_bn2");
    #pragma HLS STREAM variable=layer3_3x3_stride2_conv4_bn2 depth=16

// Layer 4
    hls::stream<acc_t> layer4_3x3_stride2("layer4_3x3_stride2");
    #pragma HLS STREAM variable=layer4_3x3_stride2 depth=16
        
    hls::stream<data_t> layer4_3x3_stride2_bn("layer4_3x3_stride2_bn");
    #pragma HLS STREAM variable=layer4_3x3_stride2_bn depth=16
    
    hls::stream<data_t> layer4_3x3_stride2_relu("layer4_3x3_stride2_relu");
    #pragma HLS STREAM variable=layer4_3x3_stride2_relu depth=16

    hls::stream<acc_t> layer4_3x3_stride2_conv2("layer4_3x3_stride2_conv2");
    #pragma HLS STREAM variable=layer4_3x3_stride2_conv2 depth=16
    
    hls::stream<data_t> layer4_3x3_stride2_conv2_bn("layer4_3x3_stride2_conv2_bn");
    #pragma HLS STREAM variable=layer4_3x3_stride2_conv2_bn depth=16

    hls::stream<acc_t> skip_layer4_3x3_stride2("skip_layer4_3x3_stride2");
    #pragma HLS STREAM variable=skip_layer4_3x3_stride2 depth=16

    hls::stream<data_t> skip_layer4_3x3_stride2_bn("skip_layer4_3x3_stride2_bn");
    #pragma HLS STREAM variable=skip_layer4_3x3_stride2_bn depth=16
    
    hls::stream<data_t> layer4_connected_path("layer4_connected_path");
    #pragma HLS STREAM variable=layer4_connected_path depth=16
    
    hls::stream<acc_t> layer4_3x3_stride2_conv3("layer4_3x3_stride2_conv3");
    #pragma HLS STREAM variable=layer4_3x3_stride2_conv3 depth=16
    
    hls::stream<data_t> layer4_3x3_stride2_conv3_bn("layer4_3x3_stride2_conv3_bn");
    #pragma HLS STREAM variable=layer4_3x3_stride2_conv3_bn depth=16
    
    hls::stream<data_t> layer4_3x3_stride2_relu_conv3("layer4_3x3_stride2_relu_conv3");
    #pragma HLS STREAM variable=layer4_3x3_stride2_relu_conv3 depth=16
    
    hls::stream<acc_t> layer4_3x3_stride2_conv4("layer4_3x3_stride2_conv4");
    #pragma HLS STREAM variable=layer4_3x3_stride2_conv4 depth=16
    
    hls::stream<data_t> layer4_3x3_stride2_conv4_bn("layer4_3x3_stride2_conv4_bn");
    #pragma HLS STREAM variable=layer4_3x3_stride2_conv4_bn depth=16

    // avg pool
    hls::stream<data_t> avg_pool_1x1_stream("avg_pool_1x1_stream");
    #pragma HLS STREAM variable=avg_pool_1x1_stream depth=16



// ============================================================================
// Layer0
// ============================================================================

// template<int CONV_OUT_C, int CONV_IN_C, int CONV_IN_H, int CONV_IN_W, int CONV_K>
conv2D<64, 3, 224, 224, 7>(layer0_input, layer0_7x7_stride2, layer0_conv1_weights_stream, 2, 3);


//BN + ReLU + MaxPool
batch_norm(
       layer0_7x7_stride2,
       layer0_7x7_stride2_bn,
       layer0_out_h, layer0_out_w, layer0_out_ch,
       layer0_conv1_bn_mean_stream,
       layer0_conv1_bn_deno_stream,
       layer0_conv1_bn_gamma_stream,
       layer0_conv1_bn_beta_stream
   );

relu(
    layer0_7x7_stride2_bn,
    layer0_7x7_stride2_relu,
    layer0_out_h * layer0_out_w * layer0_out_ch
);

max_pool(
    layer0_7x7_stride2_relu,
    layer0_7x7_stride2_mp
);

// ============================================================================
// Layer1
// ============================================================================

// template<int CONV_OUT_C, int CONV_IN_C, int CONV_IN_H, int CONV_IN_W, int CONV_K>
conv2D<64, 64, 112, 112, 3>(layer0_7x7_stride2_mp, layer1_3x3_stride1, layer1_conv_1_weights_stream, 1, 1);

batch_norm(
        layer1_3x3_stride1,
        layer1_3x3_stride1_bn,
        layer1_out_h, layer1_out_w, layer1_out_ch,
        layer1_conv1_bn_mean_stream,
        layer1_conv1_bn_deno_stream,
        layer1_conv1_bn_gamma_stream,
        layer1_conv1_bn_beta_stream
);

relu(
    layer1_3x3_stride1_bn,
    layer1_3x3_stride1_relu,
    layer1_out_h * layer1_out_w * layer1_out_ch
);

conv2D<64, 64, 112, 112, 3>(layer1_3x3_stride1_relu, layer1_3x3_stride1_conv2, layer1_conv_2_weights_stream, 1, 1);

batch_norm(
        layer1_3x3_stride1_conv2,
        layer1_3x3_stride1_conv2_bn,
        layer1_out_h, layer1_out_w, layer1_out_ch,
        layer1_conv1_bn_mean_stream,
        layer1_conv1_bn_deno_stream,
        layer1_conv1_bn_gamma_stream,
        layer1_conv1_bn_beta_stream
);

// Layer1 2nd Basic Block
conv2D<64, 64, 112, 112, 3>(layer1_3x3_stride1_conv2_bn, layer1_3x3_stride1_conv3, layer1_conv_3_weights_stream, 1, 1);

batch_norm(
        layer1_3x3_stride1_conv3,
        layer1_3x3_stride1_conv3_bn,
        layer1_out_h, layer1_out_w, layer1_out_ch,
        layer1_conv3_bn_mean_stream,
        layer1_conv3_bn_deno_stream,
        layer1_conv3_bn_gamma_stream,
        layer1_conv3_bn_beta_stream
);

relu(
    layer1_3x3_stride1_conv3_bn,
    layer1_3x3_stride1_relu_conv3,
    layer1_out_h * layer1_out_w * layer1_out_ch
);

conv2D<64, 64, 112, 112, 3>(layer1_3x3_stride1_relu_conv3, layer1_3x3_stride1_conv4, layer1_conv_4_weights_stream, 1, 1);

batch_norm(
        layer1_3x3_stride1_conv4,
        layer1_3x3_stride1_conv4_bn,
        layer1_out_h, layer1_out_w, layer1_out_ch,
        layer1_conv4_bn_mean_stream,
        layer1_conv4_bn_deno_stream,
        layer1_conv4_bn_gamma_stream,
        layer1_conv4_bn_beta_stream
);


// ============================================================================
// Layer2
// ============================================================================


// Duplicate input stream into layer1_3x3_stride1_conv4_bn1 and layer1_3x3_stride1_conv4_bn2
duplicate_stream<data_t>(layer1_3x3_stride1_conv4_bn, layer1_3x3_stride1_conv4_bn1, layer1_3x3_stride1_conv4_bn2, LAYER2_SIZE);

conv2D<128, 64, 56, 56, 3>(layer1_3x3_stride1_conv4_bn1, layer2_3x3_stride2, layer2_conv_1_weights_stream, 2, 1);

batch_norm(
        layer2_3x3_stride2,
        layer2_3x3_stride2_bn,
        layer2_out_h, layer2_out_w, layer2_out_ch,
        layer2_conv1_bn_mean_stream,
        layer2_conv1_bn_deno_stream,
        layer2_conv1_bn_gamma_stream,
        layer2_conv1_bn_beta_stream
);

relu(
    layer2_3x3_stride2_bn,
    layer2_3x3_stride2_relu,
    layer2_out_h * layer2_out_w * layer2_out_ch
);

conv2D<128, 128, 28, 28, 3>(layer2_3x3_stride2_relu, layer2_3x3_stride2_conv2, layer2_conv_2_weights_stream, 1, 1);

batch_norm(
        layer2_3x3_stride2_conv2,
        layer2_3x3_stride2_conv2_bn,
        layer2_out_h, layer2_out_w, layer2_out_ch,
        layer2_conv2_bn_mean_stream,
        layer2_conv2_bn_deno_stream,
        layer2_conv2_bn_gamma_stream,
        layer2_conv2_bn_beta_stream
);

//SKIP PATH:
conv2D<128, 64, 56, 56, 3>(layer1_3x3_stride1_conv4_bn2, skip_layer2_3x3_stride2, layer2_s_weights_stream, 2, 1);

batch_norm(
        skip_layer2_3x3_stride2,
        skip_layer2_3x3_stride2_bn,
        layer2_out_h, layer2_out_w, layer2_out_ch,
        skip_layer2_bn_mean_stream,
        skip_layer2_bn_deno_stream,
        skip_layer2_bn_gamma_stream,
        skip_layer2_bn_beta_stream
);

//Add main path + skip_path
skip_add(
    layer2_3x3_stride2_conv2_bn,
    skip_layer2_3x3_stride2_bn,
    layer2_connected_path,
    layer2_out_h * layer2_out_w * layer2_out_ch
);

//Layer2 2nd Basic Block
conv2D<128, 128, 28, 28, 3>(layer2_connected_path, layer2_3x3_stride2_conv3, layer2_conv_3_weights_stream, 1, 1);

batch_norm(
        layer2_3x3_stride2_conv3,
        layer2_3x3_stride2_conv3_bn,
        layer2_out_h, layer2_out_w, layer2_out_ch,
        layer2_conv3_bn_mean_stream,
        layer2_conv3_bn_deno_stream,
        layer2_conv3_bn_gamma_stream,
        layer2_conv3_bn_beta_stream
);

relu(
    layer2_3x3_stride2_conv3_bn,
    layer2_3x3_stride2_relu_conv3,
    layer2_out_h * layer2_out_w * layer2_out_ch
);

conv2D<128, 128, 28, 28, 3>(layer2_3x3_stride2_relu_conv3, layer2_3x3_stride2_conv4, layer2_conv_4_weights_stream, 1, 1);

batch_norm(
        layer2_3x3_stride2_conv4,
        layer2_3x3_stride2_conv4_bn,
        layer2_out_h, layer2_out_w, layer2_out_ch,
        layer2_conv4_bn_mean_stream,
        layer2_conv4_bn_deno_stream,
        layer2_conv4_bn_gamma_stream,
        layer2_conv4_bn_beta_stream
);


// ============================================================================
// Layer3
// ============================================================================


// Duplicate input stream into layer1_3x3_stride1_conv4_bn1 and layer1_3x3_stride1_conv4_bn2
duplicate_stream<data_t>(layer2_3x3_stride2_conv4_bn, layer2_3x3_stride2_conv4_bn1, layer2_3x3_stride2_conv4_bn2, LAYER3_SIZE);

//conv1
conv2D<256, 128, 28, 28, 3>(layer2_3x3_stride2_conv4_bn1, layer3_3x3_stride2, layer3_conv_1_weights_stream, 2, 1);

batch_norm(
        layer3_3x3_stride2,
        layer3_3x3_stride2_bn,
        layer3_out_h, layer3_out_w, layer3_out_ch,
        layer3_conv1_bn_mean_stream,
        layer3_conv1_bn_deno_stream,
        layer3_conv1_bn_gamma_stream,
        layer3_conv1_bn_beta_stream
);

relu(
    layer3_3x3_stride2_bn,
    layer3_3x3_stride2_relu,
    layer3_out_h * layer3_out_w * layer3_out_ch
);

//conv2
conv2D<256, 256, 14, 14, 3>(layer3_3x3_stride2_relu, layer3_3x3_stride2_conv2, layer3_conv_2_weights_stream, 1, 1);

batch_norm(
        layer3_3x3_stride2_conv2,
        layer3_3x3_stride2_conv2_bn,
        layer3_out_h, layer3_out_w, layer3_out_ch,
        layer3_conv2_bn_mean_stream,
        layer3_conv2_bn_deno_stream,
        layer3_conv2_bn_gamma_stream,
        layer3_conv2_bn_beta_stream
);

//SKIP PATH:
conv2D<256, 128, 28, 28, 3>(layer2_3x3_stride2_conv4_bn2, skip_layer3_3x3_stride2, layer3_s_weights_stream, 2, 1);

batch_norm(
        skip_layer3_3x3_stride2,
        skip_layer3_3x3_stride2_bn,
        layer3_out_h, layer3_out_w, layer3_out_ch,
        skip_layer3_bn_mean_stream,
        skip_layer3_bn_deno_stream,
        skip_layer3_bn_gamma_stream,
        skip_layer3_bn_beta_stream
);

skip_add(
    layer3_3x3_stride2_conv2_bn,
    skip_layer3_3x3_stride2_bn,
    layer3_connected_path,
    layer3_out_h * layer3_out_w * layer3_out_ch
);

//Layer3 2nd basic block
//conv3
conv2D<256, 256, 14, 14, 3>(layer3_connected_path, layer3_3x3_stride2_conv3, layer3_conv_3_weights_stream, 1, 1);

batch_norm(
        layer3_3x3_stride2_conv3,
        layer3_3x3_stride2_conv3_bn,
        layer3_out_h, layer3_out_w, layer3_out_ch,
        layer3_conv3_bn_mean_stream,
        layer3_conv3_bn_deno_stream,
        layer3_conv3_bn_gamma_stream,
        layer3_conv3_bn_beta_stream
);

relu(
    layer3_3x3_stride2_conv3_bn,
    layer3_3x3_stride2_relu_conv3,
    layer3_out_h * layer3_out_w * layer3_out_ch
);

//conv4
conv2D<256, 256, 14, 14, 3>(layer3_3x3_stride2_relu_conv3, layer3_3x3_stride2_conv4, layer3_conv_4_weights_stream, 1, 1);

batch_norm(
        layer3_3x3_stride2_conv4,
        layer3_3x3_stride2_conv4_bn,
        layer3_out_h, layer3_out_w, layer3_out_ch,
        layer3_conv4_bn_mean_stream,
        layer3_conv4_bn_deno_stream,
        layer3_conv4_bn_gamma_stream,
        layer3_conv4_bn_beta_stream
);


// ============================================================================
// Layer4
// ============================================================================


// Duplicate input stream into d_input1 and d_input2
duplicate_stream<data_t>(layer3_3x3_stride2_conv4_bn, layer3_3x3_stride2_conv4_bn1, layer3_3x3_stride2_conv4_bn2, LAYER4_SIZE);

// MAIN PATH
// template<int CONV_OUT_C, int CONV_IN_C, int CONV_IN_H, int CONV_IN_W, int CONV_K>
conv2D<512, 256, 14, 14, 3>(layer3_3x3_stride2_conv4_bn1, layer4_3x3_stride2, layer4_conv_1_weights_stream, 2, 1);

// BN + ReLU
batch_norm(
    layer4_3x3_stride2,
    layer4_3x3_stride2_bn,
    layer4_out_h, layer4_out_w, layer4_out_ch,
    layer4_conv1_bn_mean_stream,
    layer4_conv1_bn_deno_stream,
    layer4_conv1_bn_gamma_stream,
    layer4_conv1_bn_beta_stream
);

relu(
    layer4_3x3_stride2_bn,
    layer4_3x3_stride2_relu,
    layer4_out_h * layer4_out_w * layer4_out_ch
);

// Conv 3x3 stride 1
//  conv2D(
//      out_3x3_stride2_relu,
//      out_3x3_stride1,
//      conv_2_weights_stream,
//      in_h / 2, in_w / 2, out_ch, out_ch, 3, 1, 1
//  );
conv2D<512, 512, 7, 7, 3>(layer4_3x3_stride2_relu, layer4_3x3_stride2_conv2, layer4_conv_2_weights_stream, 1, 1);

// BN (no ReLU yet)
batch_norm(
    layer4_3x3_stride2_conv2,
    layer4_3x3_stride2_conv2_bn,
    layer4_out_h, layer4_out_w, layer4_out_ch,
    layer4_conv2_bn_mean_stream,
    layer4_conv2_bn_deno_stream,
    layer4_conv2_bn_gamma_stream,
    layer4_conv2_bn_beta_stream
);

// SKIP PATH:
//  conv2D(
//      d_input2,
//      skip_out_3x3_stride1,
//      conv_s_weights_stream,
//      in_h, in_w, in_ch, out_ch, 1, 2, 0
//  );
conv2D<512, 256, 14, 14, 1>(layer3_3x3_stride2_conv4_bn2, skip_layer4_3x3_stride2, layer4_s_weights_stream, 2, 1);

batch_norm(
    skip_layer4_3x3_stride1,
    skip_layer4_3x3_stride2_bn,
    layer4_out_h, layer4_out_w, layer4_out_ch,
    skip_layer4_bn_mean_stream,
    skip_layer4_bn_deno_stream,
    skip_layer4_bn_gamma_stream,
    skip_layer4_bn_beta_stream
);

// Add main_path + skip_path
skip_add(
    layer4_3x3_stride2_conv2_bn,
    skip_layer4_3x3_stride2_bn,
    layer4_connected_path,
    layer4_out_h * layer4_out_w * layer4_out_ch
);


//Layer4 2nd Basic Block
conv2D<512, 512, 7, 7, 3>(layer4_connected_path, layer4_3x3_stride2_conv3, layer4_conv_3_weights_stream, 1, 1);

batch_norm(
        layer4_3x3_stride2_conv3,
        layer4_3x3_stride2_conv3_bn,
        layer4_out_h, layer4_out_w, layer4_out_ch,
        layer4_conv3_bn_mean_stream,
        layer4_conv3_bn_deno_stream,
        layer4_conv3_bn_gamma_stream,
        layer4_conv3_bn_beta_stream
);

relu(
    layer4_3x3_stride2_conv3_bn,
    layer4_3x3_stride2_relu_conv3,
    layer4_out_h * layer4_out_w * layer4_out_ch
);

conv2D<512, 512, 7, 7, 3>(layer4_3x3_stride2_relu_conv3, layer4_3x3_stride2_conv4, layer4_conv_4_weights_stream, 1, 1);

batch_norm(
        layer4_3x3_stride2_conv4,
        layer4_3x3_stride2_conv4_bn,
        layer4_out_h, layer4_out_w, layer4_out_ch,
        layer4_conv4_bn_mean_stream,
        layer4_conv4_bn_deno_stream,
        layer4_conv4_bn_gamma_stream,
        layer4_conv4_bn_beta_stream
);

avg_pool_1x1(
    layer4_3x3_stride2_conv4_bn,    // Input stream: channels x in_h x in_w
    avg_pool_1x1_stream,   // Output stream: channels x 1 x 1
    512,          // Number of channels
    7,            // Input in_h
    7             // Input in_w
);

fully_connected(
    avg_pool_1x1_stream,        // Input vector
    output,                 // Output vector
    data_t weights[1000][512],             // Weight matrix [out_ch][in_ch]
    data_t biases[1000],                   // Bias vector [out_ch]
    512,
    1000
);

    // Write to output stream
    // for (int i = 0; i < layer4_out_h * layer4_out_w * layer4_out_ch; i++) {
    // #pragma HLS PIPELINE II=1
    //     if(!out_final.empty()) {
    //         data_t y = layer4_3x3_stride2_conv4_bn.read();
    //         output.write(y);
    //     }
    // }

}
