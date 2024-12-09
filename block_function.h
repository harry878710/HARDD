// block_function.h
#ifndef BLOCK_FUNCTION_H
#define BLOCK_FUNCTION_H

#include "resnet18.h"

void run_resnet_block(const data_t* input, data_t* output, int in_h, int in_w, int in_ch, int out_ch);

#endif
