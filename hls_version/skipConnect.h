// skipConnect.h
#ifndef SKIP_CONNECT_H
#define SKIP_CONNECT_H
#include "resnet18.h"
#include "quantize.h"

void skip_add(hls::stream<data_t> &main_path, hls::stream<data_t> &skip_path, hls::stream<data_t> &final_path, int size);

#endif
