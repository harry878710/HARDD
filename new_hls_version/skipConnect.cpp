// skipConnect.cpp
#include "skipConnect.h"

// sum = main + skip (accumulation) then clamp to 8-bit.
void skip_add(hls::stream<data_t> &main_path, hls::stream<data_t> &skip_path, hls::stream<data_t> &final_path, int size) {
    for (int i = 0; i < size; i++) {
        acc_t sum = acc_t(main_path.read()) + acc_t(skip_path.read()); // H(x) = F(x) + x
        // data_t sum = main_path[i] + skip_path[i];
        // main_path[i] = (data_t)sum; // might need to do quantization here
        final_path.write(quantize_acc_to_data(sum));
    }
}
