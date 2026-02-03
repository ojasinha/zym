#include "search.h"
#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

std::vector<std::string> find_files(const std::string& directory) {
    std::vector<std::string> files;
    
    try {
        // Recursively iterate through directory
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            // Only add regular files (not directories)
            if (entry.is_regular_file()) {
                files.push_back(entry.path().string());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error accessing directory: " << e.what() << "\n";
    }
    
    return files;
}