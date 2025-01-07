
// // Perform convolution

// #include "Conv2D.h"
// #include <iostream>
// #include <fstream>
// #include <cassert>
// #include <nlohmann/json.hpp> // JSON library
// #include <chrono>           // For timing
// #include <iomanip>          // For formatted output

// using json = nlohmann::json;
// using namespace std;

// vector<float> load_binary_data(const string &file_path, size_t size) {
//     ifstream file(file_path, ios::binary);
//     assert(file.is_open() && "Unable to open file");

//     vector<float> data(size);
//     file.read(reinterpret_cast<char *>(data.data()), size * sizeof(float));
//     file.close();

//     return data;
// }

// void conv2d_1d(const vector<float> &input, 
//                const vector<float> &kernel, 
//                const vector<float> &bias, 
//                vector<float> &output,
//                int input_height, int input_width, int input_channels,
//                int kernel_height, int kernel_width, int output_channels,
//                int stride, const string &padding, string layername) {

//     // Calculate padding
//     if (padding != "same" && padding != "valid") {
//         cerr << "Error: Unsupported padding type: " << padding << endl;
//         exit(EXIT_FAILURE);
//     }

//     int pad_height = 0, pad_width = 0;
//     if (padding == "same") {
//         pad_height = (kernel_height - 1) / 2;
//         pad_width = (kernel_width - 1) / 2;
//     }

//     // Calculate output dimensions
//     int output_height = (input_height + 2 * pad_height - kernel_height) / stride + 1;
//     int output_width = (input_width + 2 * pad_width - kernel_width) / stride + 1;
//     int output_size = output_height * output_width * output_channels;

//     // Resize the output vector
//     output.resize(output_size, 0.0f);

//     // Start measuring execution time
//     auto start_time = chrono::high_resolution_clock::now();

//     // Perform convolution
//     for (int h = 0; h < output_height; ++h) {
//         for (int w = 0; w < output_width; ++w) {
//             for (int c = 0; c < output_channels; ++c) {
//                 float value = 0.0f;

//                 for (int kh = 0; kh < kernel_height; ++kh) {
//                     for (int kw = 0; kw < kernel_width; ++kw) {
//                         for (int ic = 0; ic < input_channels; ++ic) {
//                             int ih = h * stride + kh - pad_height;
//                             int iw = w * stride + kw - pad_width;

//                             // Check bounds for valid input
//                             if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
//                                 int input_idx = ((ih * input_width + iw) * input_channels) + ic;
//                                 int kernel_idx = ((kh * kernel_width + kw) * input_channels + ic) * output_channels + c;

//                                 // assert(input_idx >= 0 && input_idx < input.size());
//                                 // assert(kernel_idx >= 0 && kernel_idx < kernel.size());

//                                 value += input[input_idx] * kernel[kernel_idx];
//                             }
//                         }
//                     }
//                 }

//                 // Add bias
//                 value += bias[c];

//                 // Store the result in the 1D output vector
//                 int output_idx = (h * output_width + w) * output_channels + c;
//                 assert(output_idx >= 0 && output_idx < output.size());
//                 output[output_idx] = value;
//             }
//         }
//     }

   
// }



//abi



#include "Conv2D.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <fstream>

void conv2d(const std::vector<float>& input, const std::vector<float>& kernel,
            const std::vector<float>& bias, std::vector<float>& output,
            const std::array<int, 4>& input_shape, const std::array<int, 4>& output_shape,
            const std::array<int, 2>& kernel_size, const std::array<int, 2>& strides,
            const std::string& padding, const std::string& layer_name) {
    int batch = input_shape[0];
    int in_height = input_shape[1], in_width = input_shape[2], in_channels = input_shape[3];
    int out_height = output_shape[1], out_width = output_shape[2], out_channels = output_shape[3];
    int kernel_height = kernel_size[0], kernel_width = kernel_size[1];

    // std::cout<<kernel.size() << " "<< batch * in_height * in_width * in_channels<<"\n";

    assert(input.size() == batch * in_height * in_width * in_channels);
    // assert(kernel.size() == kernel_height * kernel_width * in_channels * out_channels);
    assert(bias.size() == out_channels);
    assert(output.size() == batch * out_height * out_width * out_channels);
    //  std::cout<<"hhi";

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                for (int c = 0; c < out_channels; ++c) {
                    float sum = 0.0f;

                    for (int kh = 0; kh < kernel_height; ++kh) {
                        for (int kw = 0; kw < kernel_width; ++kw) {
                            for (int ic = 0; ic < in_channels; ++ic) {
                                int ih = h * strides[0] + kh;
                                int iw = w * strides[1] + kw;

                                if (padding == "same") {
                                    ih -= kernel_height / 2;
                                    iw -= kernel_width / 2;
                                }

                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int input_idx = ((b * in_height + ih) * in_width + iw) * in_channels + ic;
                                    int kernel_idx = ((kh * kernel_width + kw) * in_channels + ic) * out_channels + c;

                                    sum += input[input_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                    }

                    int output_idx = ((b * out_height + h) * out_width + w) * out_channels + c;
                    output[output_idx] = std::max(0.0f, sum + bias[c]);
                }
            }
        }
    }

    std::cout << "Conv2D Output Size = " << output.size() << std::endl;

    // std::string output_file = "F:/MultiCoreWare/C++ Application/Project_Root/data/cpp_layer_outputs/" + layer_name + ".txt";
    // std::ofstream file(output_file);

    // if (file.is_open()) {

    //     for (size_t i = 0; i < output.size(); ++i) {
    //         file << std::fixed << std::setprecision(6) << output[i] << "\n";
    //     }

    //     file.close();
    //     std::cout << "Saved layer output to " << output_file << std::endl;
    // } else {
    //     std::cerr << "Failed to open file for writing: " << output_file << std::endl;
    // }
}