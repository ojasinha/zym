#ifndef SEARCH_H
#define SEARCH_H

#include <string>
#include <vector>

// Search result
struct SearchResult {
    std::string filename;
    int line_number;
    std::string line_content;
};

// File utilities
std::vector<std::string> find_files(const std::string& directory);

// Exact pattern matching - CPU
std::vector<SearchResult> exact_search_cpu(
    const std::vector<std::string>& files,
    const std::string& pattern,
    int num_threads = 0,
    bool verbose = false
);

// Exact pattern matching - GPU
std::vector<SearchResult> exact_search_gpu(
    const std::vector<std::string>& files,
    const std::string& pattern,
    bool verbose = false
);

// Fuzzy matching - CPU
std::vector<SearchResult> fuzzy_search_cpu(
    const std::vector<std::string>& files,
    const std::string& pattern,
    int max_distance,
    int num_threads = 0,
    bool verbose = false
);

// Fuzzy matching - GPU
std::vector<SearchResult> fuzzy_search_gpu(
    const std::vector<std::string>& files,
    const std::string& pattern,
    int max_distance,
    bool verbose = false
);

#endif