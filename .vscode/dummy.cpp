#include <iostream>   
#include <direct.h> 
#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
int main() {
    // // Open the JSON file
    // // Define a buffer 
    // const size_t size = 1024; 
    // // Allocate a character array to store the directory path
    // char buffer[size];        
    
    // // Call _getcwd to get the current working directory and store it in buffer
    // if (getcwd(buffer, size) != NULL) {
    //     // print the current working directory
    //     printf("Current working directory: %s" , buffer);
    // } 
    // else {
    //     // If _getcwd returns NULL, print an error message
    //     // cerr << "Error getting current working directory" << endl;
    // }
    std::ifstream file("E:/Assignment-2-C++/Project_Root/configs/json/model_config_ragul.json");
    if (!file.is_open()) {
        std::cerr << "Failed to open the JSON file.\n";
        return 1;
    }
    // Parse the JSON file
    json config;
    try {
        file >> config;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << '\n';
        return 1;
    }
    // Ensure "layers" key exists
    if (!config.contains("layers")) {
        std::cerr << "The JSON file does not contain 'layers'.\n";
        return 1;
    }
    // Iterate through the layers
    for (const auto& layer : config["layers"]) {
        std::cout << "Layer Name: " << layer["layer_name"] << "\n";
        std::cout << "Type: " << layer["type"] << "\n";
        // Print input and output file paths
        std::cout << "Input File Path: " << layer["input_file_path"] << "\n";
        std::cout << "Output File Path: " << layer["output_file_path"] << "\n";
        // Print weights file paths
        if (layer.contains("weights_file_paths") && !layer["weights_file_paths"].empty()) {
            std::cout << "Weights File Paths:\n";
            for (const auto& path : layer["weights_file_paths"]) {
                std::cout << "  - " << path << "\n";
            }
        }
        // Print attributes
        if (layer.contains("attributes")) {
            std::cout << "Attributes:\n";
            for (const auto& [key, value] : layer["attributes"].items()) {
                std::cout << "  " << key << ": " << value << "\n";
            }
        }
        std::cout << "\n";
    }
    return 0;
}






