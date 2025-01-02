
#include "MaxPooling2D.h"
#include <algorithm>
#include <stdexcept>
#include <limits> // For std::numeric_limits

// Constructor definition
MaxPooling2D::MaxPooling2D(const std::vector<int>& inputShape, const std::vector<int>& outputShape, 
                           const std::vector<int>& strides, const std::string& padding)
    : inputShape(inputShape), outputShape(outputShape), strides(strides), padding(padding) {}

// Apply max pooling to the input tensor
void MaxPooling2D::applyPooling(const std::vector<std::vector<std::vector<float>>>& input,
                                std::vector<std::vector<std::vector<float>>>& output) {
    // Validate input dimensions
    if (input.size() != inputShape[0] || input[0].size() != inputShape[1] || input[0][0].size() != inputShape[2]) {
        throw std::invalid_argument("Input dimensions do not match the specified input shape.");
    }

    // Initialize output tensor with zeros
    output.assign(outputShape[0], std::vector<std::vector<float>>(outputShape[1], std::vector<float>(outputShape[2], 0)));

    // Perform max pooling
    for (int c = 0; c < inputShape[2]; ++c) { // Iterate over channels
        for (int i = 0; i < outputShape[0]; ++i) { // Iterate over output rows
            for (int j = 0; j < outputShape[1]; ++j) { // Iterate over output columns
                int startRow = i * strides[0];
                int startCol = j * strides[1];
                output[i][j][c] = getMaxValue(input, startRow, startCol, c, 2); // Assuming pool size of 2x2
            }
        }
    }
}

// Helper function to get the maximum value in the pooling window
float MaxPooling2D::getMaxValue(const std::vector<std::vector<std::vector<float>>>& input, 
                                int startRow, int startCol, int channel, int poolSize) {
    float maxVal = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < poolSize; ++i) {
        for (int j = 0; j < poolSize; ++j) {
            int row = startRow + i;
            int col = startCol + j;
            // Check for boundary conditions
            if (row < inputShape[0] && col < inputShape[1]) {
                maxVal = std::max(maxVal, input[row][col][channel]);
            }
        }
    }
    return maxVal;
}