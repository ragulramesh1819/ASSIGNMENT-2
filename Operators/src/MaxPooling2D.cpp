
#include "MaxPooling2D.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <fstream>
#include <iomanip>

void max_pooling2d(const std::vector<float>& input, std::vector<float>& output,
                   const std::array<int, 4>& input_shape, const std::array<int, 4>& output_shape,
                   const std::array<int, 2>& pool_size, const std::array<int, 2>& strides,
                   const std::string& padding, const std::string& layer_name) {
    int batch = input_shape[0];
    int in_height = input_shape[1], in_width = input_shape[2], in_channels = input_shape[3];
    int out_height = output_shape[1], out_width = output_shape[2];

    int pad_height = 0, pad_width = 0;

    if (padding == "same") {
        // Calculate padding based on "same" padding rule
        pad_height = std::max(0, (out_height - 1) * strides[0] + pool_size[0] - in_height);
        pad_width = std::max(0, (out_width - 1) * strides[1] + pool_size[1] - in_width);
    }

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                for (int c = 0; c < in_channels; ++c) {
                    float max_val = -std::numeric_limits<float>::infinity();

                    for (int ph = 0; ph < pool_size[0]; ++ph) {
                        for (int pw = 0; pw < pool_size[1]; ++pw) {
                            // For "valid" padding, no need to adjust indices, just ensure within bounds
                            int ih = h * strides[0] + ph;
                            int iw = w * strides[1] + pw;

                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                int input_idx = ((b * in_height + ih) * in_width + iw) * in_channels + c;
                                max_val = std::max(max_val, input[input_idx]);
                            }
                        }
                    }

                    int output_idx = ((b * out_height + h) * out_width + w) * in_channels + c;
                    output[output_idx] = max_val;
                }
            }
        }
    }

    std::cout << "MaxPool Output Size = " << output.size() << std::endl;
}
