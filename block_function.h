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
#include "test_hls_stream.h"

void run_resnet_block(const data_t* input, data_t* output, int in_h, int in_w, int in_ch, int out_ch);

#endif
