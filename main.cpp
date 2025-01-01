#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp> // For JSON parsing
#include <vector>
#include "conv2d.h"
#include "batchnormalization.h"



using json = nlohmann::json;

// Function to read binary data from a file
template <typename T>
std::vector<T> readBinaryFile(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening file: " + filePath);
    }

    // Get the file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Calculate the number of elements in the binary file (assuming each element is of type T)
    size_t numElements = fileSize / sizeof(T);

    // Read data from the file into a vector
    std::vector<T> data(numElements);
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    file.close();

    return data;
}

/// 4 d vector convertor 
std::vector<std::vector<std::vector<std::vector<float>>>> reshape_kernel(
    const std::vector<float>& kernels_flat, int num_kernels, int kernel_height, int kernel_width, int input_channels) {
    // Initialize the 4D kernel
    std::vector<std::vector<std::vector<std::vector<float>>>> kernel(
        num_kernels, std::vector<std::vector<std::vector<float>>>(
                         kernel_height, std::vector<std::vector<float>>(
                                            kernel_width, std::vector<float>(input_channels, 0))));

    int kernel_index = 0;
    for (int c = 0; c < num_kernels; ++c) {
        for (int h = 0; h < kernel_height; ++h) {
            for (int w = 0; w < kernel_width; ++w) {
                for (int ic = 0; ic < input_channels; ++ic) {
                    kernel[c][h][w][ic] = kernels_flat[kernel_index++];
                }
            }
        }
    }

    return kernel;
}




int main() {
    // Load the JSON configuration
     // Dummy input tensor (1 batch, 1 channel, 5x5 image)
      std::vector<std::vector<std::vector<float>>> input= {
        {
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
            {6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
            {11.0f, 12.0f, 13.0f, 14.0f, 15.0f},
            {16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
            {21.0f, 22.0f, 23.0f, 24.0f, 25.0f}
        }
    };
    
    std::ifstream configFile("configs/json/model_config_ragul.json");
    if (!configFile.is_open()) {
        std::cerr << "Failed to open configuration file.\n";
        return 1;
    }

    json config;
    configFile >> config;

    // Placeholder for initial input data (update with actual data if available)
    std::vector<float> inputData; // Input data for the first layer

    // Process layers sequentially
    for (const auto &layer : config["layers"]) {
        try {
            std::string layerType = layer["type"];
            if (layerType == "Conv2D" || layerType == "BatchNormalization" || layerType == "MaxPooling2D" || layerType == "Dense" || layerType == "Flatten") {
                // inputData = processLayer(layer, input); // Pass current input and get updated output

                    const json layerConfig=layer;

                    std::cout << "Processing layer: " << layerConfig["layer_name"] << " (" << layerConfig["type"] << ")\n";


                    // Perform operations based on layer type
                    std::string layerType = layerConfig["type"];
                    


                    if (layerType == "Conv2D") {
                        std::cout << "Performing Conv2D operation.\n";
                        // Conv2D logic placeholder
                        std::vector<float> Kernel_weight = readBinaryFile<float>(layerConfig["weights_file_paths"][0]);
                        std::vector<float> Bias_weight = readBinaryFile<float>(layerConfig["weights_file_paths"][1]);

                        int num_kernels =layerConfig["attributes"]["output_shape"][2];
                        int kernel_height =layerConfig["kernel_size"][0];
                        int kernel_width = layerConfig["kernel_size"][1];
                        int input_channels =layerConfig["attributes"]["input_shape"][2];

                        std::vector<std::vector<std::vector<std::vector<float>>>> weights =reshape_kernel(Kernel_weight,
                        num_kernels,kernel_height , kernel_width, input_channels);

                        input=conv2D(  input, weights, Bias_weight ,0, 0);
                    } 


                    
                    else if (layerType == "BatchNormalization") {
                        std::cout << "Performing BatchNormalization operation.\n";
                        // BatchNormalization logic placeholder
                        // Initialize BatchNormalization with 3 channels (features)
                        std::vector<int> input_shape =layerConfig["attributes"]["input_shape"];
                        std::vector<std::string> weight_file_paths=layerConfig["weights_file_paths"];


                        std::vector<std::vector<std::vector<float>>> output;

                        std::vector<float> gamma= readBinaryFile<float>(layerConfig["weights_file_paths"][0]);
                        std::vector<float> beta= readBinaryFile<float>(layerConfig["weights_file_paths"][1]);
                        std::vector<float> moving_mean= readBinaryFile<float>(layerConfig["weights_file_paths"][2]);
                        std::vector<float> moving_variance= readBinaryFile<float>(layerConfig["weights_file_paths"][3]);

                        batch_normalization( input,gamma,beta,moving_mean, moving_variance,output);
                        input=output;



                    }
                    
                    
                    // else if (layerType == "MaxPooling2D") {
                    //     std::cout << "Performing MaxPooling2D operation.\n";
                    //     // MaxPooling2D logic placeholder
                    // }
                    
                    
                    // else if (layerType == "Dense") {
                    //     std::cout << "Performing Dense operation.\n";
                    //     // Dense logic placeholder
                    // }
                    
                    
                    // else if (layerType == "Flatten") {
                    //     std::cout << "Performing Flatten operation.\n";
                    //     // Flatten logic placeholder
                    // } 
                    
                    
                    // else {
                    //     std::cerr << "Unsupported layer type: " << layerType << "\n";
                    // }
                                
            }
            
            
             else
            {
                std::cout << "Skipping unsupported layer: " << layer["layer_name"] << " (" << layerType << ")\n";
            }
        } catch (const std::exception &e)
        {
            std::cerr << "Error processing layer " << layer["layer_name"] << ": " << e.what() << "\n";
        }
    }

    std::cout << "All specified layers processed.\n";
    return 0;
}
