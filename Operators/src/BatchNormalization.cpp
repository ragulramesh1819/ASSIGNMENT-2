
#include "BatchNormalization.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <string>
#include <cassert>

// Function to perform batch normalization
void batch_normalization(const std::vector<float>& input, std::vector<float>& output,
                         const std::vector<float>& gamma, const std::vector<float>& beta,
                         const std::vector<float>& moving_mean, const std::vector<float>& moving_variance,
                         float epsilon, size_t channels, size_t height, size_t width, const std::string& layer_name) {
                            
    // Validate input sizes
    std::cout << channels * height * width <<"\n";

    // Resize output to match input size
    output.resize(input.size());

    // Perform batch normalization
    for (size_t c = 0; c < channels; ++c) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                // Calculate the index based on the assumed data format
                size_t idx = (h * width + w) * channels + c;
                assert(idx < input.size());

                // Apply batch normalization formula
                output[idx] = gamma[c] * (input[idx] - moving_mean[c]) /
                              std::sqrt(moving_variance[c] + epsilon) + beta[c];
            }
        }
    }

    // Output the size of the result for verification
    std::cout << "BatchNorm Output Size = " << output.size() << std::endl;
}

