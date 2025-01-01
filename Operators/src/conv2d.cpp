#include "conv2d.h"
#include <vector>
#include <cmath>
#include <stdexcept>

// Function to perform 2D convolution
std::vector<std::vector<std::vector<float>>> conv2D(
    const std::vector<std::vector<std::vector<float>>>& input,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& weights,
    const std::vector<float>& biases,
    int stride,
    int padding)
{
    int inputChannels = input.size();
    int inputHeight = input[0].size();
    int inputWidth = input[0][0].size();
    int outputChannels = weights.size();
    int kernelHeight = weights[0][0].size();
    int kernelWidth = weights[0][0][0].size();

    // Calculate output dimensions
    int outputHeight = std::floor((inputHeight - kernelHeight + 2 * padding) / stride) + 1;
    int outputWidth = std::floor((inputWidth - kernelWidth + 2 * padding) / stride) + 1;

    // Initialize output tensor
    std::vector<std::vector<std::vector<float>>> output(outputChannels,
        std::vector<std::vector<float>>(outputHeight, std::vector<float>(outputWidth, 0.0f)));

    // Apply padding to input if necessary
    std::vector<std::vector<std::vector<float>>> paddedInput = input;
    if (padding > 0) {
        int paddedHeight = inputHeight + 2 * padding;
        int paddedWidth = inputWidth + 2 * padding;
        paddedInput = std::vector<std::vector<std::vector<float>>>(
            inputChannels,
            std::vector<std::vector<float>>(paddedHeight, std::vector<float>(paddedWidth, 0.0f))
        );
        for (int c = 0; c < inputChannels; ++c) {
            for (int i = 0; i < inputHeight; ++i) {
                for (int j = 0; j < inputWidth; ++j) {
                    paddedInput[c][i + padding][j + padding] = input[c][i][j];
                }
            }
        }
    }

    // Perform convolution
    for (int oc = 0; oc < outputChannels; ++oc) {
        for (int oh = 0; oh < outputHeight; ++oh) {
            for (int ow = 0; ow < outputWidth; ++ow) {
                float sum = biases[oc];
                for (int ic = 0; ic < inputChannels; ++ic) {
                    for (int kh = 0; kh < kernelHeight; ++kh) {
                        for (int kw = 0; kw < kernelWidth; ++kw) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            sum += paddedInput[ic][ih][iw] * weights[oc][ic][kh][kw];
                        }
                    }
                }
                output[oc][oh][ow] = sum;
            }
        }
    }

    return output;
}
