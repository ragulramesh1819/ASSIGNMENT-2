#include "utils.h"
#include <fstream>
#include <iostream>

// Template function definition for readBinaryFile
template<typename T>
std::vector<T> readBinaryFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size % sizeof(T) != 0)
    {
        throw std::runtime_error("File size is not a multiple of data type size.");
    }

    std::vector<T> buffer(size / sizeof(T));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        throw std::runtime_error("Error reading file: " + filename);
    }

    return buffer;
}

// Function definition for load_binary_file
std::vector<float> load_binary_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        exit(1);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> data(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    file.close();

    return data;
}

// Explicit template instantiation for readBinaryFile
template std::vector<float> readBinaryFile<float>(const std::string& filename);
template std::vector<int> readBinaryFile<int>(const std::string& filename);
