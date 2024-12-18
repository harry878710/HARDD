// block_function.cpp
#include "block_function.h"
#include <hls_stream.h>
#include <ap_fixed.h>
#include <iostream>
#include "resnet18_weights.h"

// Define data types
//   typedef ap_fixed<8,4> data_t;
//   typedef ap_fixed<16,8> acc_t;

// Function to run a ResNet block
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
) {
    // Calculate output dimensions
    int out_h = in_h / 2; // stride=2 in first conv
    int out_w = in_w / 2; // out_h = out_w = 7

    // Define local streams as objects, not references
    hls::stream<data_t> d_input1("d_input1");
    #pragma HLS STREAM variable=d_input1 depth=16

    hls::stream<data_t> d_input2("d_input2");
    #pragma HLS STREAM variable=d_input2 depth=16

    hls::stream<data_t> out_3x3_stride2("out_3x3_stride2");
    #pragma HLS STREAM variable=out_3x3_stride2 depth=16

    hls::stream<data_t> out_3x3_stride2_bn("out_3x3_stride2_bn");
    #pragma HLS STREAM variable=out_3x3_stride2_bn depth=16

    hls::stream<data_t> out_3x3_stride2_relu("out_3x3_stride2_relu");
    #pragma HLS STREAM variable=out_3x3_stride2_relu depth=16

    hls::stream<data_t> out_3x3_stride1("out_3x3_stride1");
    #pragma HLS STREAM variable=out_3x3_stride1 depth=16

    hls::stream<data_t> out_3x3_stride1_bn("out_3x3_stride1_bn");
    #pragma HLS STREAM variable=out_3x3_stride1_bn depth=16

    hls::stream<data_t> skip_out_1x1_stride2("skip_out_1x1_stride2");
    #pragma HLS STREAM variable=skip_out_1x1_stride2 depth=16

    hls::stream<data_t> skip_out_1x1_stride2_bn("skip_out_1x1_stride2_bn");
    #pragma HLS STREAM variable=skip_out_1x1_stride2_bn depth=16

    hls::stream<data_t> skip_out_3x3_stride1("skip_out_3x3_stride1");
    #pragma HLS STREAM variable=skip_out_3x3_stride1 depth=16

    hls::stream<data_t> connected_path("connected_path");
    #pragma HLS STREAM variable=connected_path depth=16

    hls::stream<data_t> out_final("out_final");
    #pragma HLS STREAM variable=out_final depth=16

    // Calculate the size for duplication
    int SIZE = in_h * in_w * in_ch; // Adjust as needed

    // Duplicate input stream into d_input1 and d_input2
    duplicate_stream<data_t>(input, d_input1, d_input2, SIZE);

    // MAIN PATH
    // Conv 3x3 stride 2
     conv2D(d_input1, out_3x3_stride2, conv_1_weights_stream, in_h, in_w, in_ch, out_ch, 3, 2, 1);
    // template<int CONV_OUT_C, int CONV_IN_C, int CONV_IN_H, int CONV_IN_W, int CONV_K>
//    conv2D<512, 256, 14, 14, 3>(d_input1, out_3x3_stride2, conv_1_weights_stream, 2, 1);

    // BN + ReLU
    batch_norm(
        out_3x3_stride2,
        out_3x3_stride2_bn,
        out_h, out_w, out_ch,
        block_conv1_bn_mean_stream,
        block_conv1_bn_deno_stream,
        block_conv1_bn_gamma_stream,
        block_conv1_bn_beta_stream
    );

    relu(
        out_3x3_stride2_bn,
        out_3x3_stride2_relu,
        out_h * out_w * out_ch
    );

    // Conv 3x3 stride 1
     conv2D(
         out_3x3_stride2_relu,
         out_3x3_stride1,
         conv_2_weights_stream,
         in_h / 2, in_w / 2, out_ch, out_ch, 3, 1, 1
     );
//    conv2D<512, 512, 7, 7, 3>(out_3x3_stride2_relu, out_3x3_stride1, conv_2_weights_stream, 1, 1);

    // BN (no ReLU yet)
    batch_norm(
        out_3x3_stride1,
        out_3x3_stride1_bn,
        out_h, out_w, out_ch,
        block_conv2_bn_mean_stream,
        block_conv2_bn_deno_stream,
        block_conv2_bn_gamma_stream,
        block_conv2_bn_beta_stream
    );

    // SKIP PATH:
     conv2D(
         d_input2,
         skip_out_3x3_stride1,
         conv_s_weights_stream,
         in_h, in_w, in_ch, out_ch, 1, 2, 0
     );
//    conv2D<512, 256, 14, 14, 1>(d_input2, skip_out_3x3_stride1, conv_s_weights_stream, 2, 0);

    batch_norm(
        skip_out_3x3_stride1,
        skip_out_1x1_stride2_bn,
        out_h, out_w, out_ch,
        skip_bn_mean_stream,
        skip_bn_deno_stream,
        skip_bn_gamma_stream,
        skip_bn_beta_stream
    );

    // Add main_path + skip_path
    skip_add(
        out_3x3_stride1_bn,
        skip_out_1x1_stride2_bn,
        connected_path,
        out_h * out_w * out_ch
    );

    // Final ReLU
    relu(
        connected_path,
        out_final,
        out_h * out_w * out_ch
    );

    // Write to output stream
    for (int i = 0; i < out_h * out_w * out_ch; i++) {
    #pragma HLS PIPELINE II=1
        if(!out_final.empty()) {
            data_t y = out_final.read();
            output.write(y);
        }
    }
}

