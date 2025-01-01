// #include "batchnormalization.h"
// #include <iostream>
// #include <fstream>
// #include <stdexcept>
// #include <cmath>

// // Constructor to initialize BatchNormalization with input shape and weight file paths
// BatchNormalization::BatchNormalization(const std::vector<int>& input_shape,
//                                        const std::vector<std::string>& weight_file_paths)
//     : input_shape(input_shape), weight_file_paths(weight_file_paths) {
//     // Initialize vectors with appropriate sizes based on the input shape
//     gamma.resize(input_shape[2], 1.0f);  // 1 per channel (channel count in the 3rd dimension)
//     beta.resize(input_shape[2], 0.0f);   // 0 per channel
//     moving_mean.resize(input_shape[2], 0.0f);  // For each channel
//     moving_variance.resize(input_shape[2], 1.0f);  // For each channel
// }

// // Load weights (gamma, beta, moving_mean, moving_variance) from binary files
// void BatchNormalization::loadWeights() {
//     // Load the weights from the binary files
//     readBinaryFile(weight_file_paths[0], gamma);            // Gamma
//     readBinaryFile(weight_file_paths[1], beta);             // Beta
//     readBinaryFile(weight_file_paths[2], moving_mean);      // Moving mean
//     readBinaryFile(weight_file_paths[3], moving_variance);  // Moving variance
// }

// // Read binary data from file into a vector
// void BatchNormalization::readBinaryFile(const std::string& file_path, std::vector<float>& data) {
//     std::ifstream file(file_path, std::ios::binary);
//     if (!file) {
//         throw std::runtime_error("Error opening file: " + file_path);
//     }

//     // Get the file size and resize the vector accordingly
//     file.seekg(0, std::ios::end);
//     size_t file_size = file.tellg();
//     file.seekg(0, std::ios::beg);
//     size_t num_elements = file_size / sizeof(float);

//     data.resize(num_elements);
//     file.read(reinterpret_cast<char*>(data.data()), file_size);
//     file.close();
// }

// // Perform batch normalization on the 3D input tensor
// void BatchNormalization::forward(const std::vector<std::vector<std::vector<float>>>& input) {
//     if (input.empty() || input[0].empty() || input[0][0].empty()) {
//         throw std::invalid_argument("Input tensor is empty.");
//     }

//     // Ensure the input shape matches the specified input shape
//     if (input.size() != input_shape[0] || input[0].size() != input_shape[1] || input[0][0].size() != input_shape[2]) {
//         throw std::invalid_argument("Input shape does not match the expected shape.");
//     }

//     // Initialize the output tensor with the same dimensions as the input
//     output.resize(input_shape[0], std::vector<std::vector<float>>(input_shape[1], std::vector<float>(input_shape[2])));

//     // Apply batch normalization for each pixel across all channels
//     for (int h = 0; h < input_shape[1]; ++h) {
//         for (int w = 0; w < input_shape[2]; ++w) {
//             // Compute mean and variance across the batch dimension (input[batch][h][w])
//             float mean = 0.0f;
//             float variance = 0.0f;

//             // Compute mean
//             for (int b = 0; b < input_shape[0]; ++b) {
//                 mean += input[b][h][w];
//             }
//             mean /= input_shape[0];

//             // Compute variance
//             for (int b = 0; b < input_shape[0]; ++b) {
//                 variance += (input[b][h][w] - mean) * (input[b][h][w] - mean);
//             }
//             variance /= input_shape[0];

//             // Normalize each batch element for this pixel (h, w)
//             for (int b = 0; b < input_shape[0]; ++b) {
//                 float normalized = (input[b][h][w] - mean) / std::sqrt(variance + 1e-5f);  // Avoid division by zero
//                 output[b][h][w] = gamma[w] * normalized + beta[w];  // Apply gamma and beta
//             }
//         }
//     }
// }

// // Save the normalized output tensor to a binary file
// void BatchNormalization::saveOutput(const std::string& output_file_path) const {
//     std::ofstream file(output_file_path, std::ios::binary);
//     if (!file) {
//         throw std::runtime_error("Error opening output file: " + output_file_path);
//     }

//     // Write the output tensor to the binary file
//     size_t total_elements = output.size() * output[0].size() * output[0][0].size();
//     file.write(reinterpret_cast<const char*>(output.data()), total_elements * sizeof(float));
//     file.close();
// }

// // Getter for the normalized output tensor
// const std::vector<std::vector<std::vector<float>>>& BatchNormalization::getOutput() const {
//     return output;
// }

#include "batchnormalization.h"
#include <cmath>
#include <cassert>

// Batch Normalization function
void batch_normalization(const std::vector<std::vector<std::vector<float>>>& input,
                         const std::vector<float>& gamma, const std::vector<float>& beta,
                         const std::vector<float>& moving_mean, const std::vector<float>& moving_variance,
                         std::vector<std::vector<std::vector<float>>>& output) {
    int height = input.size();
    int width = input[0].size();
    int channels = input[0][0].size();

    // Resize output to match the input dimensions
    output.resize(height, std::vector<std::vector<float>>(width, std::vector<float>(channels, 0)));

    // Apply batch normalization
    for (int c = 0; c < channels; ++c) {
        float mean = moving_mean[c];
        float variance = moving_variance[c];
        float stddev = std::sqrt(variance + 1e-5f);  // Small epsilon for numerical stability

        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                output[h][w][c] = gamma[c] * (input[h][w][c] - mean) / stddev + beta[c];
            }
        }
    }

    // Apply ReLU activation
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                // ReLU: set negative values to 0
                output[h][w][c] = std::max(0.0f, output[h][w][c]);
            }
        }
    }
}
