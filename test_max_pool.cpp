// #include "resnet18.h"
// #include <hls_stream.h>
// #include <ap_fixed.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "max_pool.h"


int main() {
    // Initialize streams and parameters
    hls::stream<data_t> input_stream;
    hls::stream<data_t> output_stream;
    int in_height = 4;
    int in_width = 4;
    int channels = 1;

    // Example input data
    std::vector<data_t> input_data = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    // Feed data into input_stream with padding
    for(int c = 0; c < channels; c++) {
        // First row with padding
        // input_stream.write(0.0); // Left padding
        for(int i = 0; i < in_width; i++) {
            input_stream.write(input_data[c * in_height * in_width + i]);
        }
        // input_stream.write(0.0); // Right padding

        // Second row with padding
        // input_stream.write(0.0); // Left padding
        for(int i = 0; i < in_width; i++) {
            input_stream.write(input_data[c * in_height * in_width + in_width + i]);
        }
        // input_stream.write(0.0); // Right padding

        // Third row with padding
        // input_stream.write(0.0); // Left padding
        for(int i = 0; i < in_width; i++) {
            input_stream.write(input_data[c * in_height * in_width + 2 * in_width + i]);
        }
        // input_stream.write(0.0); // Right padding

        // Fourth row with padding
        // input_stream.write(0.0); // Left padding
        for(int i = 0; i < in_width; i++) {
            input_stream.write(input_data[c * in_height * in_width + 3 * in_width + i]);
        }
        // input_stream.write(0.0); // Right padding
    }

    // Call the pooling function
    max_pool(input_stream, output_stream, in_height, in_width, channels);

    // Calculate output dimensions
    int out_height = std::floor((in_height + 2 * 1 - 3) / 2) + 1; // 2
    int out_width  = std::floor((in_width + 2 * 1 - 3) / 2) + 1;  // 2

    // Read and display the output
    std::cout << "MaxPool2d Output:" << std::endl;
    for(int c = 0; c < channels; c++) {
        for(int h = 0; h < out_height; h++) {
            for(int w = 0; w < out_width; w++) {
                data_t val = output_stream.read();
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    // output file
    // ofstream outFile("output_maxpool.txt"); 
    // if (!outFile){
    //     cerr << "Error: Could not open the file for writing." << endl;
    //     return 1;
    // }
    // for (int i = 0; i < 4; i++) {
    //     outFile << output[i] << " ";
    // }
    // outFile << endl;
    // outFile.close();
    // cout << "Output written to output_maxpool.txt successfully." << endl;

    return 0;
}
