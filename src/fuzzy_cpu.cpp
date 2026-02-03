#include "search.h"
#include <fstream>
#include <thread>
#include <mutex>
#include <iostream>
#include <algorithm>

namespace {
    int levenshtein_distance(const std::string& s1, const std::string& s2) {
        int m = s1.length();
        int n = s2.length();
        
        std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
        
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= n; j++) {
            dp[0][j] = j;
        }
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s1[i-1] == s2[j-1]) {
                    dp[i][j] = dp[i-1][j-1];
                } else {
                    dp[i][j] = 1 + std::min({
                        dp[i-1][j],
                        dp[i][j-1],
                        dp[i-1][j-1]
                    });
                }
            }
        }
        
        return dp[m][n];
    }

    std::vector<SearchResult> search_file_fuzzy(
        const std::string& filepath,
        const std::string& pattern,
        int max_distance
    ) {
        std::vector<SearchResult> results;
        std::ifstream file(filepath);
        
        if (!file.is_open()) {
            return results;
        }
        
        std::string line;
        int line_number = 1;
        int pattern_len = pattern.length();
        
        while (std::getline(file, line)) {
            bool found = false;
            
            for (int window_len = pattern_len - max_distance; 
                window_len <= pattern_len + max_distance; 
                window_len++) {
                
                if (window_len <= 0) continue;
                
                for (int i = 0; i <= (int)line.length() - window_len; i++) {
                    std::string candidate = line.substr(i, window_len);
                    int dist = levenshtein_distance(pattern, candidate);
                    
                    if (dist <= max_distance) {
                        found = true;
                        break;
                    }
                }
                
                if (found) break;
            }
            
            if (found) {
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

std::vector<SearchResult> fuzzy_search_cpu(
    const std::vector<std::string>& files,
    const std::string& pattern,
    int max_distance,
    int num_threads,
    bool verbose
) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    
    if (verbose) {
        std::cout << "CPU fuzzy search: " << files.size() << " files, " 
                << num_threads << " threads, max_distance=" << max_distance << "\n";
    }
    
    std::vector<SearchResult> all_results;
    std::mutex results_mutex;
    
    auto worker = [&](int start_idx, int end_idx) {
        std::vector<SearchResult> local_results;
        
        for (int i = start_idx; i < end_idx; i++) {
            auto file_results = search_file_fuzzy(files[i], pattern, max_distance);
            local_results.insert(
                local_results.end(), 
                file_results.begin(), 
                file_results.end()
            );
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
        std::cout << "CPU fuzzy search: " << all_results.size() << " matches\n";
    }
    
    return all_results;
}
