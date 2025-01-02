#include "BatchNormalization.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>

BatchNormalization::BatchNormalization() {
    // Initialize to empty vectors
    gamma.clear();
    beta.clear();
    moving_mean.clear();
    moving_variance.clear();
    input_shape = {0, 0, 0};
    output_shape = {0, 0, 0};
}

BatchNormalization::~BatchNormalization() {}

void BatchNormalization::LoadWeights(const std::vector<std::string>& weight_paths) {
    if (weight_paths.size() != 4) {
        throw std::invalid_argument("Exactly 4 weight files must be provided (gamma, beta, moving mean, moving variance).");
    }

    // Load weights from binary files
    auto load_binary_file = [](const std::string& path) -> std::vector<float> {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Unable to open file: " + path);
        }
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<float> data(file_size / sizeof(float));
        file.read(reinterpret_cast<char*>(data.data()), file_size);
        return data;
    };

    // Load each of the weights (gamma, beta, moving_mean, moving_variance)
    gamma = load_binary_file(weight_paths[0]);
    beta = load_binary_file(weight_paths[1]);
    moving_mean = load_binary_file(weight_paths[2]);
    moving_variance = load_binary_file(weight_paths[3]);
}

void BatchNormalization::ApplyBatchNormalization(const std::vector<std::vector<std::vector<float>>>& input,
                                                  std::vector<std::vector<std::vector<float>>>& output) {
    // Apply batch normalization to the input 3D vector
    if (input.size() != input_shape[0] || input[0].size() != input_shape[1] || input[0][0].size() != input_shape[2]) {
        throw std::invalid_argument("Input shape does not match the expected shape.");
    }

    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i].resize(input[i].size());
        for (size_t j = 0; j < input[i].size(); ++j) {
            output[i][j].resize(input[i][j].size());
            for (size_t k = 0; k < input[i][j].size(); ++k) {
                // Perform batch normalization: 
                // output = gamma * ((input - moving_mean) / sqrt(moving_variance + epsilon)) + beta
                float normalized_value = (input[i][j][k] - moving_mean[k]) / std::sqrt(moving_variance[k] + 1e-5f);
                output[i][j][k] = gamma[k] * normalized_value + beta[k];
            }
        }
    }
}

void BatchNormalization::SetInputShape(const std::vector<int>& shape) {
    input_shape = shape;
}

void BatchNormalization::SetOutputShape(const std::vector<int>& shape) {
    output_shape = shape;
}

std::vector<int> BatchNormalization::GetInputShape() const {
    return input_shape;
}

std::vector<int> BatchNormalization::GetOutputShape() const {
    return output_shape;
}
