#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept>
#include<nlohmann/json.hpp>
#include <vector>
#include <cstdint>
#include "../Operators/include/Conv2D.h"
#include "../include/MaxPooling2D.h"
#include "Flatten.h"
#include "DenseLayer.h"
#include "BatchNormalization.h"
#include "utils.h"
#include <array>
using json = nlohmann::json;



int main()
{


    std::ifstream configFile("E:/Assignment-2-C++/Project_Root/configs/json/model_config_ragul.json");
    if (!configFile.is_open()) {
        std::cout<< "Failed to open configuration file.\n";
        return 1;
    }
    int c=0;
    json config;
    configFile >> config;

     std::string inputPath = "E:/Assignment-2-C++/Project_Root/resized_image_binary.bin";
    std::vector<float> input;
    input = std::vector<float>(32*32*3, 1.0); // Dummy values (1.0); 
    //  input = readBinaryFile<float>(inputPath);

    // Ouput Classes
    std::string classes[10] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

    std::vector<float> output;

    int max_index;

    // Placeholder for input and output shapes
    std::array<int, 4> input_shape{};
    std::array<int, 4> output_shape{};

    for (const auto& layer : config["layers"]) {
        std::string type = layer["type"];
        if (type == "Dropout" || type == "Flatten") continue;
        else std::cout << "Processing Layer: " << layer["layer_name"] << " (" << type << ")" << std::endl;

        if (type == "Conv2D") {

            // Load weights and bias
            auto kernel = load_binary_file(layer["weights_file_paths"][0]);
            auto bias = load_binary_file(layer["weights_file_paths"][1]);

            // Parse shapes and parameters

            input_shape = {1,layer["attributes"]["input_shape"][0],layer["attributes"]["input_shape"][1],layer["attributes"]["input_shape"][2]};

            output_shape = {1,layer["attributes"]["output_shape"][0],layer["attributes"]["output_shape"][1],layer["attributes"]["output_shape"][2]};

            std::array<int, 2> kernel_size = {layer["attributes"]["kernel_size"][0],layer["attributes"]["kernel_size"][1]};

            std::array<int, 2> strides = {layer["attributes"]["strides"][0],layer["attributes"]["strides"][1]};

            std::string padding = layer["attributes"]["padding"];

            std::cout << "InputSize = " << input.size() << std::endl;
            std::cout << "KernelSize = " << kernel.size() << std::endl;
            std::cout << "BiasSize = " << bias.size() << std::endl;

            // Allocate memory for the output tensor
            output.resize(output_shape[1] * output_shape[2] * output_shape[3]);

            // Perform the Conv2D operation
           
            conv2d(input, kernel, bias, output, input_shape, output_shape, kernel_size, strides, padding, layer["layer_name"]);
            input=output;

            std::cout << "First Channel output values "<< layer["layer_name"] <<" layer:\n";
                for (int i = 0; i < 100; ++i) {
                    std::cout << output[i] << " ";
                    if(i%32==0 && i!=0){
                        std::cout<<"\n";
                    }
                }
                std::cout << std::endl;
            
            
            std::cout << input.size() << std::endl;
            

        } 
        else if (type == "BatchNormalization") {

            // Load gamma, beta, moving_mean, and moving_variance
            auto gamma =load_binary_file( layer["weights_file_paths"][0]);
            auto beta =load_binary_file( layer["weights_file_paths"][1]);
            auto moving_mean =load_binary_file( layer["weights_file_paths"][2]);
            auto moving_variance = load_binary_file(layer["weights_file_paths"][3]);

            // Epsilon value for numerical stability
            float epsilon = 0.001;

            std::cout << "Input size: " << input.size() << std::endl;
            std::cout << "Gamma size: " << gamma.size() << std::endl;
            std::cout << "Beta size: " << beta.size() << std::endl;
            std::cout << "Moving Mean size: " << moving_mean.size() << std::endl;
            std::cout << "Moving Variance size: " << moving_variance.size() << std::endl;

            input_shape = {1,layer["attributes"]["input_shape"][0],layer["attributes"]["input_shape"][1],layer["attributes"]["input_shape"][2] };

            output_shape = {1,layer["attributes"]["output_shape"][0],layer["attributes"]["output_shape"][1],layer["attributes"]["output_shape"][2]};

            output.resize(output_shape[1] * output_shape[2] * output_shape[3]);

            // Perform the BatchNormalization operation
            batch_normalization(input, output, gamma, beta, moving_mean, moving_variance, epsilon,output_shape[3], input_shape[1], input_shape[2], layer["layer_name"]);
            input = output;


              std::cout << "First Channel output values "<< layer["layer_name"] <<" layer:\n";
                for (int i = 0; i < 100; ++i) {
                    std::cout << output[i] << " ";
                    if(i%32==0 && i!=0){
                        std::cout<<"\n";
                    }
                }
                std::cout << std::endl;

        } 
    else if (type == "MaxPooling2D") {

            // Parse shapes and parameters
            input_shape = {1,layer["attributes"]["input_shape"][0],layer["attributes"]["input_shape"][1],layer["attributes"]["input_shape"][2]};
            output_shape = {1,layer["attributes"]["output_shape"][0],layer["attributes"]["output_shape"][1],layer["attributes"]["output_shape"][2]};
            
            std::array<int, 2> pool_size = {2,2};

            std::array<int, 2> strides = { layer["attributes"]["strides"][0],layer["attributes"]["strides"][1]};
            std::string padding = layer["attributes"]["padding"];

            std::cout << "InputSize = " << input.size() << std::endl;

            // Allocate memory for the output tensor
            output.resize(output_shape[1] * output_shape[2] * output_shape[3]);

            // Perform the MaxPooling2D operation
            max_pooling2d(input, output, input_shape, output_shape, pool_size, strides, padding, layer["layer_name"]);
            input=output;


              std::cout << "First Channel output values "<< layer["layer_name"] <<" layer:\n";
                for (int i = 0; i < 100; ++i) {
                    std::cout << output[i] << " ";
                    if(i%32==0 && i!=0){
                        std::cout<<"\n";
                    }
                }
                std::cout << std::endl;

        } 
        else if (type == "Dense") {

            // Load weights and bias
            auto weights = load_binary_file(layer["weights_file_paths"][0]);
            auto bias = load_binary_file(layer["weights_file_paths"][1]);

            // Parse shapes and parameters
            std::array<int, 2> dense_input_shape = {1, layer["attributes"]["input_shape"][0]};
            std::array<int, 2> dense_output_shape = {1, layer["attributes"]["output_shape"][0] };
            std::string activation = layer["attributes"]["activation"];

            // Allocate memory for the output tensor
            output.resize(dense_output_shape[1]);

            
            // Perform the Dense operation
            dense(input, weights, bias, output, dense_input_shape, dense_output_shape, activation, layer["layer_name"]);
            input=output;

              std::cout << "First Channel output values "<< layer["layer_name"] <<" layer:\n";
                for (int i = 0; i < 10; ++i) {
                    std::cout << output[i] << " ";
                }
                std::cout << std::endl;
            
        }

        std::cout << "=======================================================================" << std::endl;
    }

    // // Display the predicted class
    auto max_elem = std::max_element(output.begin(), output.end());
    max_index = std::distance(output.begin(), max_elem);
    
    std::cout << "Max value: " << *max_elem << " at index " << max_index << std::endl;
    std::cout << "Predicted Class: " << classes[max_index] << std::endl;
    std::cout << "Processing completed successfully!" << std::endl;
    
    return 0;
}

