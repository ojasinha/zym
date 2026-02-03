#include "search.h"
#include <iostream>
#include <string>
#include <cstring>
#include <chrono>

void print_help() {
    std::cout << "zym\n\n";
    std::cout << "Usage: zym [OPTIONS] <directory> <pattern>\n\n";
    std::cout << "Options:\n";
    std::cout << "  --cpu           Use CPU implementation (default)\n";
    std::cout << "  --gpu           Use GPU implementation\n";
    std::cout << "  --fuzzy <N>     Use fuzzy matching with max distance N\n";
    std::cout << "  --threads <N>   Number of CPU threads (default: auto)\n";
    std::cout << "  --verbose       Show detailed execution info\n";
    std::cout << "  --time          Show execution time\n";
    std::cout << "  --help          Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  zym /path/to/code TODO\n";
    std::cout << "  zym --gpu /path/to/code main\n";
    std::cout << "  zym --fuzzy 2 /path/to/code FIXME\n";
    std::cout << "  zym --gpu --fuzzy 1 --verbose /path/to/code TODO\n";
}

int main(int argc, char* argv[]) {
    bool use_gpu = false;
    bool fuzzy = false;
    int max_distance = 1;
    int num_threads = 0;
    bool verbose = false;
    bool show_time = false;
    std::string directory;
    std::string pattern;
    
    // Parse arguments
    int positional_count = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_help();
            return 0;
        } else if (arg == "--cpu") {
            use_gpu = false;
        } else if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--fuzzy") {
            fuzzy = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                max_distance = std::atoi(argv[++i]);
            }
        } else if (arg == "--threads") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --threads requires a number\n";
                return 1;
            }
            num_threads = std::atoi(argv[++i]);
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--time" || arg == "-t") {
            show_time = true;
        } else if (arg[0] != '-') {
            if (positional_count == 0) {
                directory = arg;
            } else if (positional_count == 1) {
                pattern = arg;
            } else {
                std::cerr << "Error: Too many positional arguments\n";
                print_help();
                return 1;
            }
            positional_count++;
        } else {
            std::cerr << "Error: Unknown option: " << arg << "\n";
            print_help();
            return 1;
        }
    }
    
    if (positional_count != 2) {
        std::cerr << "Error: Missing required arguments\n";
        print_help();
        return 1;
    }
    
    // Find files
    auto start_total = std::chrono::high_resolution_clock::now();
    
    auto files = find_files(directory);
    if (files.empty()) {
        std::cerr << "Error: No files found in " << directory << "\n";
        return 1;
    }
    
    if (verbose) {
        std::cout << "Found " << files.size() << " files\n";
        std::cout << "Mode: " << (use_gpu ? "GPU" : "CPU") 
                  << ", " << (fuzzy ? "fuzzy" : "exact") << "\n";
        if (fuzzy) {
            std::cout << "Max distance: " << max_distance << "\n";
        }
    }
    
    // Execute search
    auto start_search = std::chrono::high_resolution_clock::now();
    std::vector<SearchResult> results;
    
    if (fuzzy) {
        if (use_gpu) {
            results = fuzzy_search_gpu(files, pattern, max_distance, verbose);
        } else {
            results = fuzzy_search_cpu(files, pattern, max_distance, num_threads, verbose);
        }
    } else {
        if (use_gpu) {
            results = exact_search_gpu(files, pattern, verbose);
        } else {
            results = exact_search_cpu(files, pattern, num_threads, verbose);
        }
    }
    
    auto end_search = std::chrono::high_resolution_clock::now();
    auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_search - start_search);
    
    // Print results
    for (const auto& result : results) {
        std::cout << result.filename << ":" << result.line_number 
                  << ":" << result.line_content << "\n";
    }
    
    // Summary
    if (verbose || show_time) {
        auto end_total = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total);
        
        std::cout << "\n";
        std::cout << "Matches: " << results.size() << "\n";
        std::cout << "Search time: " << search_duration.count() << " ms\n";
        std::cout << "Total time: " << total_duration.count() << " ms\n";
    }
    
    return 0;
}
