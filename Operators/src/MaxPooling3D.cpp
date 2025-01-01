// #include "MaxPooling3D.h"
// #include <algorithm>
// #include <iostream>
// #include <limits>

// MaxPooling3D::MaxPooling3D(int poolHeight, int poolWidth, int stride)
//     : poolHeight(poolHeight), poolWidth(poolWidth), stride(stride) {}

// std::vector<std::vector<std::vector<float>>> MaxPooling3D::applyMaxPooling(const std::vector<std::vector<std::vector<float>>>& input) {
//     int inputHeight = input.size();                  // Height of the input feature map
//     int inputWidth = input[0].size();                 // Width of the input feature map
//     int inputChannels = input[0][0].size();           // Channels (Depth)

//     // Calculate output dimensions
//     int outputHeight = (inputHeight - poolHeight) / stride + 1;
//     int outputWidth = (inputWidth - poolWidth) / stride + 1;
    
//     // Initialize the output tensor with the calculated dimensions
//     std::vector<std::vector<std::vector<float>>> output(outputHeight, 
//                                                         std::vector<std::vector<float>>(outputWidth, 
//                                                         std::vector<float>(inputChannels, 0)));

//     // Apply max pooling
//     for (int c = 0; c < inputChannels; ++c) {
//         for (int i = 0; i < outputHeight; ++i) {
//             for (int j = 0; j < outputWidth; ++j) {
//                 float maxVal = -std::numeric_limits<float>::infinity();
                
//                 // Iterate through the pooling window
//                 for (int m = 0; m < poolHeight; ++m) {
//                     for (int n = 0; n < poolWidth; ++n) {
//                         int row = i * stride + m;
//                         int col = j * stride + n;

//                         if (row < inputHeight && col < inputWidth) {
//                             maxVal = std::max(maxVal, input[row][col][c]);
//                         }
//                     }
//                 }
                
//                 // Assign the max value to the corresponding position in the output
//                 output[i][j][c] = maxVal;
//             }
//         }
//     }

//     return output;
// }


// max_pooling2d.cpp
#include "max_pooling2d.h"
#include <algorithm>
#include <stdexcept>

MaxPooling2D::MaxPooling2D(const std::vector<int>& inputShape, const std::vector<int>& outputShape, 
                           const std::vector<int>& strides, const std::string& padding)
    : inputShape(inputShape), outputShape(outputShape), strides(strides), padding(padding) {}

void MaxPooling2D::applyPooling(const std::vector<std::vector<std::vector<float>>>& input,
                                std::vector<std::vector<std::vector<float>>>& output) {
    if (input.size() != inputShape[0] || input[0].size() != inputShape[1] || input[0][0].size() != inputShape[2]) {
        throw std::invalid_argument("Input dimensions do not match the specified input shape.");
    }

    output.resize(outputShape[0], std::vector<std::vector<float>>(outputShape[1], std::vector<float>(outputShape[2], 0)));

    for (int c = 0; c < inputShape[2]; ++c) {
        for (int i = 0; i < outputShape[0]; ++i) {
            for (int j = 0; j < outputShape[1]; ++j) {
                int startRow = i * strides[0];
                int startCol = j * strides[1];
                output[i][j][c] = getMaxValue(input, startRow, startCol, c, 2); // Pool size = 2
            }
        }
    }
}

float MaxPooling2D::getMaxValue(const std::vector<std::vector<std::vector<float>>>& input, 
                                int startRow, int startCol, int channel, int poolSize) {
    float maxVal = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < poolSize; ++i) {
        for (int j = 0; j < poolSize; ++j) {
            int row = startRow + i;
            int col = startCol + j;
            if (row < inputShape[0] && col < inputShape[1]) {
                maxVal = std::max(maxVal, input[row][col][channel]);
            }
        }
    }
    return maxVal;
}
