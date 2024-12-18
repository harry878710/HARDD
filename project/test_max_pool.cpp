#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "resnet18.h"
#include "max_pool.h"
#include <hls_stream.h>

int main() {
    // Initialize streams
    hls::stream<data_t> input_stream;
    hls::stream<data_t> output_stream;

    int in_height = 4;
    int in_width = 4;
    int channels = 64;

    // Prepare input data
    // Input:
    //  1   2   3   4
    //  5   6   7   8
    //  9  10  11  12
    // 13  14  15  16
    std::cout << "MaxPool2d Input:" << std::endl;
    data_t val = 1;
    for(int h = 0; h < in_height; h++) {
        for(int w = 0; w < in_width; w++) {
            std::cout << val << " ";
            for(int c = 0; c < channels; c++) {
                // data_t val = (h+1)*0.01 + (w+1)*0.001;
                input_stream.write(val);
            }
            val++;
        }
        std::cout << std::endl;
    }

    // Call the pooling function
    max_pool(input_stream, output_stream);

    // Calculate output dimensions
    int out_height = std::floor((in_height + 2*1 - 3)/2) + 1;
    int out_width  = std::floor((in_width + 2*1 - 3)/2) + 1;

    std::ofstream outFile("output_maxpool.txt"); 
    if (!outFile){
        std::cerr << "Error: Could not open the file for writing." << std::endl;
        return 1;
    }

    // Read and display the output
    std::cout << "MaxPool2d Output:" << std::endl;
    for(int h = 0; h < out_height; h++) {
        for(int w = 0; w < out_width; w++) {
            data_t val_L0 = 0;
            for(int c = 0; c < channels; c++) {
                val_L0 = output_stream.read();
            }
            std::cout << val_L0 << " ";
            outFile << val_L0 << " ";
        }
        std::cout << std::endl;
        outFile << std::endl;
    } 
    outFile.close();
    std::cout << "Output written to output_maxpool.txt successfully." << std::endl;

    return 0;
}


// // #include "resnet18.h"
// // #include <hls_stream.h>
// // #include <ap_fixed.h>
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <cmath>
// #include "resnet18.h"
// #include "max_pool.h"
// // using namespace std;

// int main() {
//     // Initialize streams and parameters
//     hls::stream<data_t> input_stream;
//     hls::stream<data_t> output_stream;
//     int in_height = 4;
//     int in_width = 4;
//     int channels = 1;

//     // input
//     std::cout << "MaxPool2d input:" << std::endl;
//     for(int h = 0; h < 4; h++) {
//         for(int w = 0; w < 4; w++) {
//             data_t val = (h+1)*(w+1);  // Unique value per position
//             std::cout << val << " ";
//             for(int c = 0; c < 1; c++) {
//                 input_stream.write(val);
//             }
//         }
//         std::cout << std::endl;
//     }

//     // Call the pooling function
//     max_pool(input_stream, output_stream);

//     // Calculate output dimensions
//     int out_height = std::floor((in_height + 2 * 1 - 3) / 2) + 1; // 56
//     int out_width  = std::floor((in_width + 2 * 1 - 3) / 2) + 1;  // 56

//     // output file
//     std::ofstream outFile("output_maxpool.txt"); 
//     if (!outFile){
//         std::cerr << "Error: Could not open the file for writing." << std::endl;
//         return 1;
//     }
//     // Read and display the output
//     std::cout << "MaxPool2d Output:" << std::endl;
//     for(int h = 0; h < out_height; h++) {
//         for(int w = 0; w < out_width; w++) {
//             data_t val_L0 = 0.0;
//             for(int c = 0; c < channels; c++) {
//                 data_t val = output_stream.read();
//                 val_L0 = val;
//             }
//             std::cout << val_L0 << " ";
//             outFile << val_L0 << " ";
//         }
//         std::cout << std::endl;
//         outFile << std::endl;
//     } 
//     outFile << std::endl;
//     outFile.close();
//     std::cout << "Output written to output_maxpool.txt successfully." << std::endl;

//     return 0;
// }
