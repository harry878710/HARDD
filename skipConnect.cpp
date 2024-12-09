// skipConnect.cpp
#include "skipConnect.h"

// sum = main + skip (accumulation) then clamp to 8-bit.
void skip_add(data_t *main_path, data_t *skip_path, int size) {
    for (int i = 0; i < size; i++) {
        acc_t sum = (acc_t)main_path[i] + (acc_t)skip_path[i]; // H(x) = F(x) + x
        main_path[i] = (data_t)sum; // might need to do quantization here
    }
}
