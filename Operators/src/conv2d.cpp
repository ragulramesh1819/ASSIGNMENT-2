#include "Conv2D.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cmath>

Conv2D::Conv2D(const std::vector<int>& inputShape, const std::vector<int>& outputShape, 
               const std::vector<int>& kernelSize, const std::vector<int>& strides, 
               const std::string& padding, const std::string& activation)
    : inputShape(inputShape), outputShape(outputShape), kernelSize(kernelSize),
      strides(strides), padding(padding), activation(activation) {}

void Conv2D::loadWeights(const std::string& kernelFile, const std::string& biasFile) {
    std::ifstream kernelStream(kernelFile, std::ios::binary);
    std::ifstream biasStream(biasFile, std::ios::binary);

    if (!kernelStream.is_open() || !biasStream.is_open()) {
        throw std::runtime_error("Failed to open weight files.");
    }

    int outputChannels = outputShape[2];
    int inputChannels = inputShape[2];
    kernel.resize(outputChannels, 
                  std::vector<std::vector<std::vector<float>>>(inputChannels, 
                  std::vector<std::vector<float>>(kernelSize[0], 
                  std::vector<float>(kernelSize[1]))));

    for (int o = 0; o < outputChannels; ++o) {
        for (int i = 0; i < inputChannels; ++i) {
            for (int r = 0; r < kernelSize[0]; ++r) {
                kernelStream.read(reinterpret_cast<char*>(kernel[o][i][r].data()), 
                                  kernelSize[1] * sizeof(float));
            }
        }
    }

    bias.resize(outputChannels);
    biasStream.read(reinterpret_cast<char*>(bias.data()), outputChannels * sizeof(float));
}

std::vector<std::vector<std::vector<float>>> Conv2D::padInput(
    const std::vector<std::vector<std::vector<float>>>& input) {
    if (padding != "same") return input;

    int padHeight = (outputShape[0] - 1) * strides[0] + kernelSize[0] - inputShape[0];
    int padWidth = (outputShape[1] - 1) * strides[1] + kernelSize[1] - inputShape[1];
    int padTop = padHeight / 2, padLeft = padWidth / 2;

    std::vector<std::vector<std::vector<float>>> paddedInput(
        inputShape[0] + padHeight, 
        std::vector<std::vector<float>>(inputShape[1] + padWidth, 
                                        std::vector<float>(inputShape[2], 0)));

    for (size_t h = 0; h < inputShape[0]; ++h) {
        for (size_t w = 0; w < inputShape[1]; ++w) {
            for (size_t c = 0; c < inputShape[2]; ++c) {
                paddedInput[h + padTop][w + padLeft][c] = input[h][w][c];
            }
        }
    }

    return paddedInput;
}

std::vector<std::vector<std::vector<float>>> Conv2D::forward(
    const std::vector<std::vector<std::vector<float>>>& input) {
    auto paddedInput = padInput(input);
    int outputHeight = outputShape[0];
    int outputWidth = outputShape[1];
    int outputChannels = outputShape[2];

    std::vector<std::vector<std::vector<float>>> output(outputHeight, 
        std::vector<std::vector<float>>(outputWidth, std::vector<float>(outputChannels, 0)));

    for (int oh = 0; oh < outputHeight; ++oh) {
        for (int ow = 0; ow < outputWidth; ++ow) {
            for (int oc = 0; oc < outputChannels; ++oc) {
                for (int ic = 0; ic < inputShape[2]; ++ic) {
                    for (int kh = 0; kh < kernelSize[0]; ++kh) {
                        for (int kw = 0; kw < kernelSize[1]; ++kw) {
                            int ih = oh * strides[0] + kh;
                            int iw = ow * strides[1] + kw;
                            output[oh][ow][oc] += paddedInput[ih][iw][ic] * kernel[oc][ic][kh][kw];
                        }
                    }
                }
                output[oh][ow][oc] += bias[oc];
            }
        }
    }

    applyActivation(output);
    return output;
}

void Conv2D::applyActivation(std::vector<std::vector<std::vector<float>>>& output) {

    if (activation == "relu") {
        for (auto& row : output) {
            for (auto& col : row) {
                for (auto& val : col) {
                    val = std::max(0.0f, val);
                }
            }
        }
        
    }
}
