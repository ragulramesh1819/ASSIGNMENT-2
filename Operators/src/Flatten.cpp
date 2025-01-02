#include "Flatten.h"
#include <stdexcept>
#include <iostream>

Flatten::Flatten() {}

void Flatten::SetInputShape(const std::vector<int>& shape) {
    if (shape.size() != 3) {
        throw std::invalid_argument("Input shape must be a 3D vector.");
    }
    input_shape_ = shape;
}

void Flatten::SetOutputShape(const std::vector<int>& shape) {
    if (shape.size() != 1) {
        throw std::invalid_argument("Output shape must be a 1D vector.");
    }
    output_shape_ = shape;
}

void Flatten::ApplyFlatten(const std::vector<std::vector<std::vector<float>>>& input,
                           std::vector<float>& output) {
    // Validate input shape
    if (input.size() != input_shape_[0] || 
        input[0].size() != input_shape_[1] || 
        input[0][0].size() != input_shape_[2]) {
        throw std::invalid_argument("Input dimensions do not match the specified input shape.");
    }

    // Calculate the total size of the output vector
    int total_size = input_shape_[0] * input_shape_[1] * input_shape_[2];

    if (total_size != output_shape_[0]) {
        throw std::logic_error("Flatten operation dimensions mismatch between input and output.");
    }

    // Flatten the input 3D vector into a 1D vector
    output.resize(total_size);
    int index = 0;
    for (int i = 0; i < input_shape_[0]; ++i) {
        for (int j = 0; j < input_shape_[1]; ++j) {
            for (int k = 0; k < input_shape_[2]; ++k) {
                output[index++] = input[i][j][k];
            }
        }
    }
    
}
