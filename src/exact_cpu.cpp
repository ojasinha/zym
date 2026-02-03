#include "search.h"
#include <fstream>
#include <thread>
#include <mutex>
#include <iostream>

namespace {
    std::vector<SearchResult> search_file(
        const std::string& filepath, 
        const std::string& pattern
    ) {
        std::vector<SearchResult> results;
        std::ifstream file(filepath);
        
        if (!file.is_open()) {
            return results;
        }
        
        std::string line;
        int line_number = 1;
        
        while (std::getline(file, line)) {
            if (line.find(pattern) != std::string::npos) {
                SearchResult result;
                result.filename = filepath;
                result.line_number = line_number;
                result.line_content = line;
                results.push_back(result);
            }
            line_number++;
        }
        
        file.close();
        return results;
    }
}

std::vector<SearchResult> exact_search_cpu(
    const std::vector<std::string>& files,
    const std::string& pattern,
    int num_threads,
    bool verbose
) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    
    if (verbose) {
        std::cout << "CPU exact search: " << files.size() << " files, " 
                  << num_threads << " threads\n";
    }
    
    std::vector<SearchResult> all_results;
    std::mutex results_mutex;
    
    auto worker = [&](int start_idx, int end_idx) {
        std::vector<SearchResult> local_results;
        
        for (int i = start_idx; i < end_idx; i++) {
            auto file_results = search_file(files[i], pattern);
            local_results.insert(local_results.end(), 
                                file_results.begin(), 
                                file_results.end());
        }
        
        std::lock_guard<std::mutex> lock(results_mutex);
        all_results.insert(
            all_results.end(), 
            local_results.begin(), 
            local_results.end()
        );
    };
    
    std::vector<std::thread> threads;
    int files_per_thread = files.size() / num_threads;
    
    for (int i = 0; i < num_threads; i++) {
        int start = i * files_per_thread;
        int end = (i == num_threads - 1) ? files.size() : (i + 1) * files_per_thread;
        threads.emplace_back(worker, start, end);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    if (verbose) {
        std::cout << "CPU exact search: " << all_results.size() << " matches\n";
    }
    
    return all_results;
}
