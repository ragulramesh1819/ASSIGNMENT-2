#include <iostream>
#include <fstream>
#include <string>
#include<nlohmann/json.hpp>
#include <vector>
#include "../Operators/include/Conv2D.h"
#include "../include/MaxPooling2D.h"
#include "Flatten.h"
#include "DenseLayer.h"
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



int main() {
    //Load the JSON configuration

      std::vector<std::vector<std::vector<float>>> input=std::vector<std::vector<std::vector<float>>>(32, 
                        std::vector<std::vector<float>>(32, std::vector<float>(3, 2.0)) // Dummy values (1.0)
                    ); 

       std::vector<float> input_flatten;   
    std::ifstream configFile("E:/Assignment-2-C++/Project_Root/configs/json/model_config_ragul.json");
    if (!configFile.is_open()) {
        std::cout<< "Failed to open configuration file.\n";
        return 1;
    }
    int c=0;
    json config;
    configFile >> config;

    //Process layers sequentially

    for (const auto &layer : config["layers"]) {
        try {

            std::string layerType = layer["type"];
            if (layerType == "Conv2D" || layerType == "BatchNormalization" || layerType == "MaxPooling2D" || layerType == "Dense" || layerType == "Flatten") {
                // inputData = processLayer(layer, input); // Pass current input and get updated output

                    const json layerConfig=layer;

                    std::cout << "Processing layer: " << layerConfig["layer_name"] << " (" << layerConfig["type"] << ")\n";


                    // Perform operations based on layer type
                    std::string layerType = layerConfig["type"];
                    // std::cout<<layerType<<std::endl;
                    


                    if (layerType == "Conv2D") {
                        std::cout << "Performing Conv2D operation.\n";
                    

                         // Define layer parameters
                        //  std :: cout << "inside conv2d" << std::endl;
                        std::vector<int> inputShape = layerConfig["attributes"]["input_shape"];      // Height, Width, Channels
                        std::vector<int> outputShape = layerConfig["attributes"]["output_shape"];    // Height, Width, Channels
                        std::vector<int> kernelSize = layerConfig["attributes"]["kernel_size"];      // Kernel dimensions
                        std::string Kernel_weight =layerConfig["weights_file_paths"][0];
                        std::string Bias_weight = layerConfig["weights_file_paths"][1];
                        std::vector<int> strides = {1, 1};                                           // Strides
                        std::string padding = "same";                                                // Padding type
                        std::string activation = "relu";  
                        Conv2D convLayer(inputShape, outputShape, kernelSize, strides, padding, activation);
                        convLayer.loadWeights(Kernel_weight,Bias_weight);

                        std::vector<std::vector<std::vector<float>>> output;
                        output = convLayer.forward(input);
                        input=output;
                    
                    } 


                    
                    else if (layerType == "BatchNormalization") {

                        std::cout << "Performing BatchNormalization operation.\n";


                        // Initialize the batch normalization layer
                        BatchNormalization batchNorm;

                        // Load weights from files
                        batchNorm.LoadWeights(layerConfig["weights_file_paths"]);

                        // Set input and output shapes
                        batchNorm.SetInputShape(layerConfig["attributes"]["input_shape"]);
                        batchNorm.SetOutputShape(layerConfig["attributes"]["output_shape"]);

                         std::vector<std::vector<std::vector<float>>> output;

                        // Apply batch normalization
                        batchNorm.ApplyBatchNormalization(input, output);
                        input =output;



                    }
                    
                    
                    else if (layerType == "MaxPooling2D") {
                        std::cout << "Performing MaxPooling2D operation.\n";
    

                       // Define expected shapes and parameters
                        std::vector<int> inputShape = layerConfig["attributes"]["input_shape"];      // [height, width, channels]
                        std::vector<int> outputShape = layerConfig["attributes"]["output_shape"];    // [height, width, channels]
                        std::vector<int> strides =layerConfig["attributes"]["strides"];              // [stride_height, stride_width]
                        std::string padding = layerConfig["attributes"]["padding"];

                        // Instantiate the MaxPooling2D layer
                        MaxPooling2D maxPoolingLayer(inputShape, outputShape, strides, padding);

                        // Prepare the output tensor (3D vector)
                        std::vector<std::vector<std::vector<float>>> output;

                        // Apply max pooling
                        maxPoolingLayer.applyPooling(input, output);
                        input=output;
                       

                    }
                      else if (layerType == "Flatten") {
                        std::cout << "Performing Flatten operation.\n";
                        // Dense logic placeholder


                        std::vector<int> input_shape = layerConfig["attributes"]["input_shape"];
                        std::vector<int> output_shape =layerConfig["attributes"]["output_shape"];

                        // Initialize Flatten layer
                        Flatten flatten;
                        flatten.SetInputShape(input_shape);
                        flatten.SetOutputShape(output_shape);
                        std::vector<float> output;
                        flatten.ApplyFlatten(input, output);
                        input_flatten=output;


                    }
                    
                    
                    
                    else if (layerType == "Dense") {
                        std::cout << "Performing Dense operation.\n";
                        
                        std::string weights_file = layerConfig["weights_file_paths"][0];
                        std::string bias_file =layerConfig["weights_file_paths"][1];
                         std::string activation =layerConfig["attributes"]["activation"];

                        // Define the input and output shapes
                        std::vector<int> input_shape =layerConfig["attributes"]["input_shape"];
                        std::vector<int> output_shape =layerConfig["attributes"]["output_shape"] ;

                       
                        // Create a DenseLayer object
                        DenseLayer dense_layer(input_flatten, weights_file, bias_file, input_shape, output_shape, activation);

                        //Perform forward pass
                        dense_layer.forward();

                        //Get and print the output of the DenseLayer
                        input_flatten = dense_layer.get_output();
                    }

                    else {
                        std::cerr << "Unsupported layer type: " << layerType << "\n";
                    }
                                
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
}



    