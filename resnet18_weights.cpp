#include "ResNet18_weights.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

// External declarations of weights and BN parameters.
extern data_t block_conv1_weights[BLOCK_CONV1_SIZE];
//extern acc_t block_conv1_bias[BLOCK_CONV1_OUT_C];
extern float   block_conv1_bn_mean[OUT_C];
extern float   block_conv1_bn_deno[OUT_C];
extern float   block_conv1_bn_gamma[OUT_C];
extern float   block_conv1_bn_beta[OUT_C];

extern data_t block_conv2_weights[BLOCK_CONV2_SIZE];
//extern acc_t block_conv2_bias[BLOCK_CONV2_OUT_C];
extern float   block_conv2_bn_mean[OUT_C];
extern float   block_conv2_bn_deno[OUT_C];
extern float   block_conv2_bn_gamma[OUT_C];
extern float   block_conv2_bn_beta[OUT_C];

extern data_t skip_conv_weights[SKIP_CONV_SIZE];
//extern acc_t skip_conv_bias[SKIP_CONV_OUT_C];
extern float   skip_bn_mean[OUT_C];
extern float   skip_bn_deno[OUT_C];
extern float   skip_bn_gamma[OUT_C];
extern float   skip_bn_beta[OUT_C];


// Function to Extract Data from resnet_parameter.txt File
void extract_data_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << "\n";
        return;
    }

    std::string line;
    int index = 0;

    if (std::getline(file, line) == "conv1")
   // Read Block Conv1 Parameters
   case(layers)
    for (int i = 0; i < block_conv1_weights.size() && std::getline(file, line); i++) {
        block_conv1_weights[i] = std::stof(line);
    }
    for (int i = 0; i < block_conv1_bias.size() && std::getline(file, line); i++) {
        block_conv1_bias[i] = std::stof(line);
    }
    for (int i = 0; i < block_conv1_bn_mean.size() && std::getline(file, line); i++) {
        block_conv1_bn_mean[i] = std::stof(line);
    }
    for (int i = 0; i < block_conv1_bn_deno.size() && std::getline(file, line); i++) {
        block_conv1_bn_deno[i] = std::stof(line);
    }
    for (int i = 0; i < block_conv1_bn_gamma.size() && std::getline(file, line); i++) {
        block_conv1_bn_gamma[i] = std::stof(line);
    }
    for (int i = 0; i < block_conv1_bn_beta.size() && std::getline(file, line); i++) {
        block_conv1_bn_beta[i] = std::stof(line);
    }

    // Read Block Conv2 Parameters
    for (int i = 0; i < block_conv2_weights.size() && std::getline(file, line); i++) {
        block_conv2_weights[i] = std::stof(line);
    }
    for (int i = 0; i < block_conv2_bias.size() && std::getline(file, line); i++) {
        block_conv2_bias[i] = std::stof(line);
    }
    for (int i = 0; i < block_conv2_bn_mean.size() && std::getline(file, line); i++) {
        block_conv2_bn_mean[i] = std::stof(line);
    }
    for (int i = 0; i < block_conv2_bn_deno.size() && std::getline(file, line); i++) {
        block_conv2_bn_deno[i] = std::stof(line);
    }
    for (int i = 0; i < block_conv2_bn_gamma.size() && std::getline(file, line); i++) {
        block_conv2_bn_gamma[i] = std::stof(line);
    }
    for (int i = 0; i < block_conv2_bn_beta.size() && std::getline(file, line); i++) {
        block_conv2_bn_beta[i] = std::stof(line);
    }

    // Read Skip Connection Parameters
    for (int i = 0; i < skip_conv_weights.size() && std::getline(file, line); i++) {
        skip_conv_weights[i] = std::stof(line);
    }
    for (int i = 0; i < skip_conv_bias.size() && std::getline(file, line); i++) {
        skip_conv_bias[i] = std::stof(line);
    }
    for (int i = 0; i < skip_bn_mean.size() && std::getline(file, line); i++) {
        skip_bn_mean[i] = std::stof(line);
    }
    for (int i = 0; i < skip_bn_deno.size() && std::getline(file, line); i++) {
        skip_bn_deno[i] = std::stof(line);
    }
    for (int i = 0; i < skip_bn_gamma.size() && std::getline(file, line); i++) {
        skip_bn_gamma[i] = std::stof(line);
    }
    for (int i = 0; i < skip_bn_beta.size() && std::getline(file, line); i++) {
        skip_bn_beta[i] = std::stof(line);
    }

    file.close();
}
float   skip_bn_beta[OUT_C] = {
    0.0f
};